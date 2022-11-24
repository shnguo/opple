import os
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import*
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import datetime as dt
#%%
data = pd.read_excel('进单历史数据_v20221122.xlsx',header=1).fillna(0)
data = data.iloc[:-1,1:-3]
df_agg = data.groupby('物料代码_加密').agg('sum')
sku = np.array(['sku'+str(i+1) for i in range(len(df_agg))])
wuliao_raw = df_agg.index
new_df = []
for t in df_agg.columns:
    jindan = df_agg[t].values
    month = np.array([str(t) for i in range(len(df_agg))])
    new_data = np.stack((month,sku,jindan))
    new_df.append(new_data)
new_df = np.hstack(new_df)
new_df = new_df.T
new_df = pd.DataFrame(new_df,columns=['date','sku','jindan'])
new_df['jindan'] = new_df['jindan'].astype('float')
new_df['date'] = pd.to_datetime(new_df['date'], format="%Y%m")
new_df["time_idx"] = new_df["date"].dt.year * 12 + new_df["date"].dt.month
new_df["time_idx"] -= new_df["time_idx"].min()
new_df["month"] = new_df.date.dt.month.astype(str).astype("category")
new_df["log_volume"] = np.log(new_df.jindan + 1e-8)
#%%
max_prediction_length = 2
max_encoder_length = 12
training_cutoff = new_df["time_idx"].max() - max_prediction_length
training = TimeSeriesDataSet(
    new_df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="jindan",
    group_ids=["sku"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=2,
    max_prediction_length=max_prediction_length,
    static_categoricals=["sku"],
    static_reals=[],
    time_varying_known_categoricals=["month"],
    # variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
    variable_groups= {},
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=['jindan','log_volume'],
    target_normalizer=GroupNormalizer(
        groups=["sku"], transformation="softplus"
    ),  # use softplus and normalize by group
    # target_normalizer=TorchNormalizer(transformation='softplus'),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)
validation = TimeSeriesDataSet.from_dataset(training, new_df, predict=True, stop_randomization=True)
batch_size = 32  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
#%%
pl.seed_everything(42)
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=30, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard
trainer = pl.Trainer(
    max_epochs=200,
    # gpus=1,
    accelerator='gpu',
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=30,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.3,
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
#%%
best_model_path = trainer.checkpoint_callback.best_model_path
#'lightning_logs\\lightning_logs\\version_0\\checkpoints\\epoch=182-step=5490.ckpt'
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
#%%
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
(actuals - predictions).abs().mean()
#tensor(1.9939)
#%%
# raw_prediction, x = best_tft.predict(
#     training.filter(lambda x: (x.sku == "sku2") & (x.time_idx_first_prediction == 17)),
#     mode="quantiles",
#     return_x=True,
# )
#%%
encoder_data = new_df[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]
last_data = new_df[lambda x: x.time_idx == x.time_idx.max()]
decoder_data = pd.concat(
    [last_data.assign(date=lambda x: x.date + pd.offsets.MonthBegin(i)) for i in range(1, max_prediction_length + 1)],
    ignore_index=True,
)
decoder_data["time_idx"] = decoder_data["date"].dt.year * 12 + decoder_data["date"].dt.month
decoder_data["time_idx"] += encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min()
new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
#%%
import time
pred = []
start_time=time.time()
for sku_ in sku:
    new_raw_predictions, new_x = best_tft.predict(
        new_prediction_data[new_prediction_data['sku'] == sku_], mode="quantiles", return_x=True
    )
    pre =torch.squeeze(new_raw_predictions).detach().numpy().reshape(1,-1)
    pred.append(pre)
    print(sku_)
end_time=time.time()
print(end_time-start_time)
#1585.0193243026733s
#%%
columns=[]
for m in ['202211','202212']:
    for q in [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]:
        c_name = m + '_Q' + str(int(q*100))
        columns.append(c_name)
prediction = np.array(pred).squeeze()
prediction = pd.DataFrame(prediction, columns=columns)
prediction = prediction.astype('float64')
df_agg = df_agg.reset_index()
concat_df = pd.concat([df_agg,prediction],axis=1)
concat_df.to_csv('result/销量预测_20221112_物料代码聚合train1.csv',encoding="utf_8_sig")
#%%
all_pred, all_x = best_tft.predict(new_prediction_data, mode="raw", return_x=True)









