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
    'data/1208_new.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)

data = []
for column in data_raw.columns:
    data.append(
        go.Scatter(
            x=data_raw.index, y=data_raw[column], mode='markers', name=column))

# data = [
#     go.Scatter(
#         x=data_raw.index,
#         y=data_raw['VNC31蓄电池A电压'],
#         mode='markers',
#         name='VNC31蓄电池A电压')
# ]

plotly.offline.plot(data, filename='Satillite.html', auto_open=True)
