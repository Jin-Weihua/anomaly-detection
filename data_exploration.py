#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_exploration.py
@Time    :   2018/12/01 20:19:19
@Author  :   靳卫华 
@Version :   1.0
@Contact :   wh.jin@hotmail.com
@Desc    :   None
'''

# access to system parameters
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

# import data
# 列名字有中文的时候，encoding='utf-8',不然会出错
# index_col设置属性列，parse_dates设置是否解析拥有时间值的列
dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
data_raw = pd.read_csv(
    'data/data.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)
print(data_raw.info())
# print(data_raw.sample(10))
data_raw.drop([
    'TNT1-Y太阳电池内板温度1', 'TNT14锂离子蓄电池组A温度1', 'TNT15锂离子蓄电池组A温度2',
    'TNT4-Y太阳电池内板温度4'],axis=1,inplace=True)
print('Train columns with null values:\n', data_raw.isnull().sum())
print("-" * 10)

print(data_raw.describe(include='all'))

data_new = data_raw.dropna()
data_new.to_csv('data/1208_new.csv', encoding='utf-8')
"""
绘制多个子图
一个Figure对象可以包含多个子图（Axes），在matplotlib中用Axes对象表示一个绘图区域，称为子图，可以使用subplot()快速绘制包含多个子图的图表，它的调用形式如下：
subplot(numRows,numCols,plotNum)
图表的整个绘图区域被等分为numRows行和numCols列，然后按照从左到下的顺序对每个区域进行编号，左上区域的编号为1。plotNum参数指定创建的Axes对象所在的区域
"""
x_columns = [
    'INA1_PCU输出母线电流', 'INA4_A电池组充电电流', 'INA2_A电池组放电电流', 'INZ6_-Y太阳电池阵电流',
    'INZ7_+Y太阳电池阵电流', 'VNC31蓄电池A电压', 'VNZ3MEA电压(BCDR)', 'VNZ4A组蓄电池BEA信号',
    'INA4_A电池组充电电流'
]

# x_columns = ['VNC31蓄电池A电压', 'VNZ3MEA电压(BCDR)', 'VNZ4A组蓄电池BEA信号']
# # y_columns = ['INZ7_VC1_+Y太阳电池阵电流(VC1)']
# y_columns = ['INA4_A电池组充电电流']

# fig = plt.figure(figsize=(15, 6))
# plt.title(data_raw.columns[1], FontProperties=font)
# plt.plot(data_raw.loc[:, ['INA1_PCU输出母线电流']], '-')
# plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))
# # plt.xticks(pd.date_range('2016/11/27 12:16:00', '2016/11/28 16:44:59'),rotation=30)#设置时间标签显示格式
# plt.xticks(rotation=30)
# plt.show()

fig1 = plt.figure(figsize=(15, 70), dpi=200)
for i, column in enumerate(data_new.columns):
    ax = fig1.add_subplot(len(data_new.columns), 1, i + 1)
    ax.plot(data_raw.loc[:, [column]], 'o', color='green', markersize=0.5)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))

    # plt.axes([0.14, 0.35, 0.77, 0.9])
    # ax.xaxis.set_tick_params(rotation=30)
    ax.set_title(column, FontProperties=font)
plt.subplots_adjust(
    left=0.02, bottom=0.01, right=0.98, top=0.99, hspace=0.4, wspace=0.3)
# left=0.02, bottom=0.01, right=0.98, top=0.99 分别代表到画布的左侧和底部的距离占整幅图宽和高的比例
plt.savefig('result/column_new.png')

print('finish')