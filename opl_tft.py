import os
import copy
import time
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import datetime
import pytorch_forecasting
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE,SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import argparse
from ch_input import df_to_ch
from uuid import uuid4, UUID
from log import get_logger
from tqdm import tqdm
import gc
import warnings
import json
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')
logger = get_logger(os.path.basename(__file__))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',
                        type=str,
                        default='./data/202101-202303.csv',
                        help='input file')
    parser.add_argument('--uuid',
                        type=str,
                        default=str(datetime.date.today()),
                        help='uuid')
    parser.add_argument('--forecast',
                        type=int,
                        default=4,
                        help='forecast month')
    parser.add_argument('--pricefile',
                        type=str,
                        default='./data/pricefile.csv',
                        help='input price file')
    opt = parser.parse_args()
    return opt


def pre_process(df):
    df = df.rename(columns={
        'ymonth': 'year_month',
        'date': 'datetime',
        'item_code': 'unique_id',
        'cml_or_qty': 'y',
        'amount_total':'mount'
    })
    nan_category_col = df.columns[df.isna().all()].tolist()
    category_col = list(set(df.columns)-set(['year_month','datetime','unique_id','y','mount','unit_price'])-set(nan_category_col))
    category_col=['big_class_id', 'pdt', 'second_pdt', 
                  'cat_id', 'pro_series', 'office',
                   ]
    df.drop(columns=nan_category_col,inplace=True)
    for c in  category_col+['unique_id']:
        df[c] = df[c].astype(str)
    logger.info(f'category_col={category_col}')
    df['y'].fillna(0,inplace=True)
    df['mount'].fillna(0,inplace=True)
    df = df.groupby(['year_month','unique_id']+category_col, as_index=False).agg({'y':'sum', 'mount':'sum'})
    # df['y'] = df['y'].apply(lambda x: x if x else 1e-8)
    df['price'] = df['mount'] / (df['y']+1e-8)
    # df['price'] = df.apply(lambda row: row['unit_price'] if row['unit_price'] else row['mount'] / row['y'],axis=1)
    df['year_month'] = pd.to_datetime(df['year_month'], format='%Y%m')
    df['year'] = df['year_month'].dt.year
    df['month'] = df['year_month'].dt.month
    # print(df.head())
    df_year = pd.DataFrame({'year':df['year'].unique(),'j':-1})
    df_month =  pd.DataFrame({'month':range(1,13),'j':-1})
    df_id = df[['unique_id']+category_col].drop_duplicates()
    df_id['j']=-1
    # df_id = pd.DataFrame({'unique_id':df['unique_id'].unique(),'j':-1})
    df_year_month = pd.merge(df_year,df_month)
    df_year_month = df_year_month[~((df_year_month['year']==df['year'].max())&(df_year_month['month']>df['year_month'].max().month))]
    df_year_month_id = pd.merge(df_year_month,df_id)[['year','month','unique_id']+category_col]
    print(f'df_year_month_id={len(df_year_month_id)}')
    df = pd.merge(df_year_month_id,df,how='left')
    df['y'].fillna(0,inplace=True)
    df['mount'].fillna(0,inplace=True)
    df['year_month'] = df.apply(lambda x:datetime.date(x['year'],x.month,1),axis=1)
    df = df.sort_values(by=['unique_id','year_month'])
    df = df.groupby(['unique_id']+category_col, as_index=False).apply(lambda group: group.ffill())
    df = df.groupby(['unique_id']+category_col, as_index=False).apply(lambda group: group.bfill())
    df = df.fillna('unkown')
    df["time_idx"] = df["year"] * 12 + df["month"]
    df["time_idx"] -= df["time_idx"].min()
    df["month"] = df["month"].astype(str).astype("category")
    df["year"] = df["year"].astype(str).astype("category")
    df['unique_id'] = df['unique_id'].astype(str).astype("category")
    df['y'] = df['y'].astype('float')
    df["log_y"] = np.log(df.y + 1e-8)
    df["avg_y_by_id"] = df.groupby(["unique_id"],
                                   observed=True).y.transform("mean")
    for c in category_col:
        df[c] = df[c].astype(str).astype("category")
        df[f'avg_y_by_{c}'] = df.groupby([c],
                                   observed=True).y.transform("mean")
    print(len(df),len(df['unique_id'].unique()))
    return df,category_col


def bulid_data_loader(data, forecast,category_col):
    max_prediction_length = forecast
    max_encoder_length = forecast * 4
    training_cutoff = data["time_idx"].max() - max_prediction_length
    filter_date = training_cutoff-max_encoder_length//2
    id_min_idx = data[data['y']>0].groupby(['unique_id']).time_idx.min()
    # print(id_min_idx[id_min_idx>filter_date].index)
    print(data.shape)
    data = data[~data['unique_id'].isin(id_min_idx[id_min_idx>filter_date].index)]
    print(data.shape)
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="y",
        group_ids=["unique_id"],
        min_encoder_length=max_encoder_length //2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["unique_id"]+category_col,
        time_varying_known_categoricals=["month"],
        time_varying_known_reals=["time_idx", "price"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "y",
            "log_y",
            "avg_y_by_id",
        ]+[f'avg_y_by_{c}' for c in category_col],
        target_normalizer=GroupNormalizer(
            groups=["unique_id"],
            transformation="softplus"),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        # categorical_encoders={'year':pytorch_forecasting.data.encoders.NaNLabelEncoder(add_nan=True)}
        # allow_missing_timesteps=True
    )
    training_full = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="y",
        group_ids=["unique_id"],
        min_encoder_length=max_encoder_length //2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["unique_id"]+category_col,
        time_varying_known_categoricals=["month"],
        time_varying_known_reals=["time_idx", "price"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "y",
            "log_y",
            "avg_y_by_id",
        ]+[f'avg_y_by_{c}' for c in category_col],
        target_normalizer=GroupNormalizer(
            groups=["unique_id"],
            transformation="softplus"),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        # categorical_encoders={'year':pytorch_forecasting.data.encoders.NaNLabelEncoder(add_nan=True)}
        # allow_missing_timesteps=True
    )
    validation = TimeSeriesDataSet.from_dataset(training_full, data, predict=True, stop_randomization=True)
    # create dataloaders for model
    batch_size = 128 # set this between 32 to 128
    train_dataloader = training_full.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    return data,training,train_dataloader,val_dataloader

def baseline_model(val_dataloader):
    baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
    return MAE()(baseline_predictions.output, baseline_predictions.y)
    # actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    # baseline_predictions = Baseline().predict(val_dataloader).cpu()
    return (actuals - baseline_predictions).abs().mean().item()

def train_step_1(training,train_dataloader,val_dataloader):
    # configure network and trainer
    pl.seed_everything(42)
    trainer = pl.Trainer(
        gpus=0,
        # clipping gradients is a hyperparameter and important to prevent divergance
        # of the gradient for recurrent neural networks
        gradient_clip_val=0.1,
    )


    tft = TemporalFusionTransformer.from_dataset(
        training,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=0.01,
        hidden_size=16,  # most important hyperparameter apart from learning rate
        # number of attention heads. Set to up to 4 for large datasets
        attention_head_size=2,
        dropout=0.1,  # between 0.1 and 0.3 are good values
        hidden_continuous_size=8,  # set to <= hidden_size
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        # reduce learning rate if no improvement in validation loss after x epochs
        reduce_on_plateau_patience=4,
        optimizer='adam',
        log_interval=-1
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
    # find optimal learning rate
    from lightning.pytorch.tuner import Tuner
    res = Tuner(trainer).lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()

def train(training,train_dataloader,val_dataloader):
    pl.seed_everything(42)
        # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=20, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard
    if torch.cuda.is_available():
        accelerator='gpu'
    else:
        accelerator='cpu'
    trainer = pl.Trainer(
        max_epochs=300,
        # max_epochs=3,
        accelerator=accelerator, 
        # devices=0,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=30,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        log_interval=-1,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    )
    return trainer

def forcast_train(best_tft,df_filter,val_dataloader,horizon):
    predictions = best_tft.predict(val_dataloader, mode="raw", return_index=True,trainer_kwargs=dict(accelerator="cpu"))
    forcast_train_list = []
    data_length = len(predictions.index)
    for index in tqdm(range(data_length)):
        uniq_id = predictions.index.iloc[index]['unique_id']
        tmp_df = df_filter[df_filter.unique_id == uniq_id][-horizon:]
        tmp_df['y_array']=[x for x in np.array(predictions.output.prediction)[index].tolist()]
        tmp_df['y']=np.array(predictions.output.prediction)[index,:,3]
        tmp_df['mount'] = tmp_df['y']*tmp_df['price']
        tmp_df['year'] = tmp_df['year'].astype('int')
        tmp_df['month'] = tmp_df['month'].astype('int')
        tmp_df['unique_id'] = tmp_df['unique_id'].astype(str)
        tmp_df['model'] = 'TFT'
        forcast_train_list.append(tmp_df)
    forcast_train_df = pd.concat(forcast_train_list,ignore_index=True)
    return forcast_train_df
            



def forcast_future(best_tft,df_filter,forecast_length,price_df=None):
    # select last 24 months from data (max_encoder_length is 24)
    max_encoder_length=forecast_length*4
    max_prediction_length = forecast_length
    encoder_data = df_filter[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

    # select last known data point and create decoder data from it by repeating it and incrementing the month
    # in a real world dataset, we should not just forward fill the covariates but specify them to account
    # for changes in special days and prices (which you absolutely should do but we are too lazy here)
    last_data = df_filter[lambda x: x.time_idx == x.time_idx.max()]
    decoder_data = pd.concat(
        [last_data.assign(year_month=lambda x: x.year_month + pd.offsets.MonthBegin(i)) for i in range(1, max_prediction_length + 1)],
        ignore_index=True,
    )

    # add time index consistent with "data"
    decoder_data["time_idx"] = decoder_data["year_month"].dt.year * 12 + decoder_data["year_month"].dt.month
    decoder_data["time_idx"] += encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min()

    # adjust additional time feature(s)
    decoder_data["month"] = decoder_data.year_month.dt.month.astype(str).astype("category")  # categories have be strings
    decoder_data['year_month'] = decoder_data['year_month'].dt.date
    # combine encoder and decoder data
    new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
    if len(price_df)>0:
        new_prediction_data = pd.merge(new_prediction_data,price_df,how='left',left_on=['unique_id','year','month'],right_on=['unique_id','year','month'],suffixes=('', '_y'))
        new_prediction_data['price'] = new_prediction_data.apply(lambda row:row['unit_price'] if row['unit_price']>0 else row['price'],axis=1)
    predictions = best_tft.predict(new_prediction_data, mode="raw",return_index=True,trainer_kwargs=dict(accelerator="cpu"))
    data_length = len(predictions.index)
    forcast_future_list = []
    for index in tqdm(range(data_length)):
        uniq_id = predictions.index.iloc[index]['unique_id']
        tmp_df = new_prediction_data[new_prediction_data.unique_id == uniq_id][-forecast_length:]
        tmp_df['y_array']=[x for x in np.array(predictions.output.prediction)[index].tolist()]
        tmp_df['y']=np.array(predictions.output.prediction)[index,:,3]
        tmp_df['mount'] = tmp_df['y']*tmp_df['price']
        tmp_df['year'] = tmp_df['year'].astype('int')
        tmp_df['month'] = tmp_df['month'].astype('int')
        tmp_df['unique_id'] = tmp_df['unique_id'].astype(str)
        tmp_df['model'] = 'TFT'
        forcast_future_list.append(tmp_df)
    forcast_future_df = pd.concat(forcast_future_list,ignore_index=True)
    return forcast_future_df



def ori_data_to_ch(df_full,_datetime,_uuid=None):
    df_full_copy = df_full.rename(columns={
        'ymonth': 'year_month',
        'date': 'date',
        'item_code': 'unique_id',
        'cml_or_qty': 'y',
        'amount_total':'mount',
        'item_name':'description'
    })
    df_full_copy['y'] = df_full_copy['y'].apply(lambda x: x if x>0 else 1e-8)
    df_full_copy['mount'] = df_full_copy['mount'].apply(lambda x: x if x>0 else 1e-8)
    df_full_copy['price'] = df_full_copy.apply(lambda row: row['unit_price'] if row['unit_price']>0 else row['mount'] / row['y'],axis=1)
    df_full_copy['year_month'] = df_full_copy['year_month'].astype('str')
    df_full_copy['date'] = df_full_copy['date'].astype('str')
    df_full_copy['unique_id'] = df_full_copy['unique_id'].astype('str')
    df_full_copy['y'] = df_full_copy['y'].astype('float')
    
    # print(df_full_copy.dtypes)
    df_to_ch(df_full_copy,columns=[
             'unique_id', 'date', 'year_month', 'y', 'price', 'mount','description'
         ],_uuid=_uuid,timestamp=_datetime,table='opl_ori_data')

def pre_data_to_ch(df_full,_datetime,_uuid=None):
    df_full_copy = df_full.copy()
    df_full_copy['year'] = df_full_copy['year'].astype('int')
    df_full_copy['month'] = df_full_copy['month'].astype('int')
    df_full_copy['unique_id'] = df_full_copy['unique_id'].astype(str)
    df_to_ch(df_full_copy,columns=[
             'unique_id', 'year', 'month', 'year_month', 'y', 'price', 'mount'
         ],
         table='opl_pre_data_month',timestamp=_datetime)

def pre_price_data(df):
    df = df.rename(columns={
        'ymonth': 'year_month',
        'item_code': 'unique_id',
    })
    df['year_month'] = pd.to_datetime(df['year_month'], format='%Y%m')
    df['year'] = df['year_month'].dt.year
    df['month'] = df['year_month'].dt.month
    df["month"] = df["month"].astype(str).astype("category")
    df["year"] = df["year"].astype(str).astype("category")
    df['unique_id'] = df['unique_id'].astype(str).astype("category")
    df.drop_duplicates(subset=['unique_id',"month","year"],inplace=True)
    return df


def main(_uuid,file,forecast_length=4,pricefile=None):
    global logger
    time_start=time.time()
    print('start')
    # _uuid = str(uuid4())
    _datetime = datetime.datetime.now()
    logger.info(f'_datetime={_datetime}')
    _uuid = _uuid
    logger.info(f'_uuid={_uuid}')
    df_full = pd.read_csv(file)
    logger.info('Insert opl_ori_data')
    try:
        ori_data_to_ch(df_full,_uuid=_uuid,_datetime=_datetime)
    except Exception as e:
        logger.error(e)
    df_full,category_col = pre_process(df_full)
    logger.info('Insert opl_pre_data_month')
    try:
        pre_data_to_ch(df_full,_datetime=_datetime)
    except Exception as e:
        logger.error(e)   
    df_filter,training,train_dataloader,val_dataloader =  bulid_data_loader(df_full, forecast_length,category_col)
    # print(f'baseline:{baseline_model(val_dataloader)}')
    # train_step_1(training,train_dataloader,val_dataloader)
    trainer = train(training,train_dataloader,val_dataloader)
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    print('Begin to forcast training data')
    forcast_train_df = forcast_train(best_tft,df_filter,val_dataloader,forecast_length)
    print('Begin to insert training forcast data')
    try:
        df_to_ch(forcast_train_df,columns=[
                'unique_id', 'year', 'month', 'year_month', 'y','y_array', 'price', 'mount','model'
            ],
            _type='val',
            table='opl_forcasting_month',_uuid=_uuid,timestamp=_datetime)
    except Exception as e:
        logger.error(e)
    print('Begin to forcast future data')
    price_df=''
    if pricefile:
        price_df = pd.read_csv(pricefile)
        price_df = pre_price_data(price_df)
    forcast_future_df = forcast_future(best_tft,df_filter,forecast_length,price_df)
    print('Begin to insert future forcast data')
    try:
        df_to_ch(forcast_future_df,columns=[
             'unique_id', 'year', 'month', 'year_month', 'y', 'y_array','price', 'mount','model'
         ],
         _type='future',
         table='opl_forcasting_month',_uuid=_uuid,timestamp=_datetime)
    except Exception as e:
        logger.error(e)
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
    return forcast_train_df,forcast_future_df



if __name__ == '__main__':
    opt = parse_opt()
    main(opt.uuid,opt.file,opt.forecast,opt.pricefile)
