from clickhouse_driver import connect
from clickhouse_driver import Client
import datetime
from uuid import uuid4, UUID
import pandas as pd

_test_df = pd.DataFrame({
    'year': ['2023', '2023', '2023', '2023', '2023', '2023', '2023', '2023'],
    'month': ['3', '4', '5', '6', '3', '4', '5', '6'],
    'unique_id': [
        '2E6531FE6B2B0D83DC76D60379178F40', '2E6531FE6B2B0D83DC76D60379178F40',
        '2E6531FE6B2B0D83DC76D60379178F40', '2E6531FE6B2B0D83DC76D60379178F40',
        '2E6531FE6B2B0D83DC76D60379178F40', '2E6531FE6B2B0D83DC76D60379178F40',
        '2E6531FE6B2B0D83DC76D60379178F40', '2E6531FE6B2B0D83DC76D60379178F40'
    ],
    'year_month': [
        datetime.date(2023, 3, 1),
        datetime.date(2023, 4, 1),
        datetime.date(2023, 5, 1),
        datetime.date(2023, 6, 1),
        datetime.date(2023, 3, 1),
        datetime.date(2023, 4, 1),
        datetime.date(2023, 5, 1),
        datetime.date(2023, 6, 1)
    ],
    'y': [
        7.156797376950408e-09, 5.819647697080654e-08, 1.451308122568662e-07,
        5.5812968469126645e-08, 7.156797376950408e-09, 5.819647697080654e-08,
        1.451308122568662e-07, 5.5812968469126645e-08
    ],
    'y_json':None,
    'mount': [75.37, 75.37, 75.37, 75.37, 75.37, 75.37, 75.37, 75.37],
    'price': [75.37, 75.37, 75.37, 75.37, 75.37, 75.37, 75.37, 75.37],
    'time_idx': [26, 27, 28, 29, 26, 27, 28, 29],
    'log_y': [
        -18.420680743952367, -18.420680743952367, -18.420680743952367,
        -18.420680743952367, -18.420680743952367, -18.420680743952367,
        -18.420680743952367, -18.420680743952367
    ],
    'avg_y_by_id': [
        0.11538461538461539, 0.11538461538461539, 0.11538461538461539,
        0.11538461538461539, 0.11538461538461539, 0.11538461538461539,
        0.11538461538461539, 0.11538461538461539
    ]
})


def df_to_ch(df,
             _uuid=True,
             columns=None,
             _type=None,
             timestamp=datetime.datetime.now(),
             table=''):

    df_copy = df[columns].copy()
    if _uuid:
        df_copy['id'] = _uuid
    df_copy['timestamp'] = timestamp
    if _type:
        df_copy['type'] = _type
    # print(df_copy)
    with Client(host='192.168.0.48',
                database='opple_tsdb',
                user='admin',
                password='123456',
                # port=8123,
                settings={
                    'use_numpy': True,
                }) as client:
        client.insert_dataframe(f'INSERT INTO {table} VALUES', df_copy)
        client.execute(f'OPTIMIZE TABLE {table} final')


# df_to_ch(_test_df,
#          columns=[
#              'unique_id', 'year', 'month', 'year_month', 'y','y_json', 'price', 'mount'
#          ],
#          _type='test',
#          table='opl_forcasting_month')
