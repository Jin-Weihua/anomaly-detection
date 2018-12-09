#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_plot.py
@Time    :   2018/12/08 16:29:35
@Author  :   靳卫华 
@Version :   1.0
@Contact :   wh.jin@hotmail.com
@Desc    :   None
'''

# here put the import lib
import sys
print("Python version: {}".format(sys.version))

# collection of functions for data processing and analysis
# modeled after R dataframes with SQL like features
import pandas as pd
print("pandas version: {}".format(pd.__version__))

# collection of functions for scientific and publication-ready visualization
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.font_manager import FontProperties
print("matplotlib version: {}".format(matplotlib.__version__))

# foundational package for scientific computing
import numpy as np
print("NumPy version: {}".format(np.__version__))

# ignore warnings
# import warnings
# warnings.filterwarnings('ignore')
print('-' * 25)

font = FontProperties(fname='/Library/Fonts/Songti.ttc')
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
mpl.rcParams['axes.titlesize'] = 7
mpl.rc('xtick', labelsize=5)  # 设置坐标轴刻度显示大小
mpl.rc('ytick', labelsize=5)
font_size = 5

dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
data_raw = pd.read_csv(
    'data/1208_new.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)
print(data_raw.info())

fig1 = plt.figure(figsize=(15, 70), dpi=200)
for i, column in enumerate(data_raw.columns):
    ax = fig1.add_subplot(len(data_raw.columns), 1, i + 1)
    ax.plot(data_raw.loc[:, [column]], 'o', color='green', markersize=0.5)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))
    ax.set_title(column, FontProperties=font)
plt.subplots_adjust(
    left=0.02, bottom=0.01, right=0.98, top=0.99, hspace=0.4, wspace=0.3)
# left=0.02, bottom=0.01, right=0.98, top=0.99 分别代表到画布的左侧和底部的距离占整幅图宽和高的比例
plt.savefig('result/plot.png')