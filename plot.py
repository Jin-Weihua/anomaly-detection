#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   plot.py
@Time    :   2018/12/08 18:31:57
@Author  :   靳卫华 
@Version :   1.0
@Contact :   wh.jin@hotmail.com
@Desc    :   None
'''

# here put the import lib
import plotly
plotly.tools.set_credentials_file(
    username='wh.jin', api_key='DADs4nY8rboe9LuL6Iuz')
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd

dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
data_raw1 = pd.read_csv(
    'data/data_scaler.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

data_prd1 = pd.read_csv(
    'data/LstmAutoEncoder7_prd.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

data_raw = data_raw1.iloc[0:80]
data_prd = data_prd1.iloc[0:80]
# data_plot = data_raw.iloc[96000:]

data = []
print(type(data_raw.index))
column = 'INZ12_BBDR2输出电流' 
data.append(go.Scatter(x=data_raw.index, y=data_raw[column], mode='markers+lines', name=column))
data.append(go.Scatter(x=data_prd.index, y=data_prd[column], mode='markers', name=column))


# for column in data_raw.columns:
#     data.append(
#         go.Scatter(
#             x=data_plot.index, y=data_plot[column], mode='markers', name=column))


plotly.offline.plot(data, filename='result/test.html', auto_open=True)
