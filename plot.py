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
data_raw = pd.read_csv(
    'data/data_scaler.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

data_prd = pd.read_csv(
    'data/LstmAutoEncoder1_prd.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

# data_plot = data_raw.iloc[96000:]

data = []
column = 'VNC9蓄电池A单体9电压' 
data.append(go.Scatter(x=data_raw.index, y=data_raw.iloc[0:200][column], mode='markers', name=column))
data.append(go.Scatter(x=data_raw.index, y=data_prd.iloc[0:200][column], mode='markers', name=column))

# for column in data_raw.columns:
#     data.append(
#         go.Scatter(
#             x=data_plot.index, y=data_plot[column], mode='markers', name=column))


plotly.offline.plot(data, filename='result/column.html', auto_open=True)
