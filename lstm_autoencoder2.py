#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   lstm_autoencoder6.py
@Time    :   2018/12/10 19:22:10
@Author  :   靳卫华 
@Version :   1.0
@Contact :   wh.jin@hotmail.com
@Desc    :   None
'''
##########################
# 只使用全连接Dense，利用9个特征，并且对'INA1_PCU输出母线电流'平滑
##########################

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from library.plot_utils import visualize_reconstruction_error
from library.auto_encoder import LstmAutoEncoder2

DO_TRAINING = True


def main():
    data_dir_path = 'data'
    model_dir_path = 'model/LstmAutoEncoder2'
    dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    satellite_data1 = pd.read_csv(
        data_dir_path + '/data_std.csv',
        sep=',',
        index_col=0,
        encoding='utf-8',
        parse_dates=True,
        date_parser=dateparser)
    column = ['INA4_A电池组充电电流','INA2_A电池组放电电流','TNZ1PCU分流模块温度1','INZ6_-Y太阳电池阵电流','VNA2_A蓄电池整组电压','VNC1_蓄电池A单体1电压','VNZ2MEA电压(S3R)','VNZ4A组蓄电池BEA信号']
    INA1_PCU = satellite_data1['INA1_PCU输出母线电流'].rolling(5).mean()
    concat_data = pd.concat([satellite_data1.loc[:,column], INA1_PCU], axis=1).dropna()
    satellite_data = concat_data.iloc[0:96700]#96700
    print(satellite_data.head())
    satellite_np_data = satellite_data.as_matrix()
    scaler = MinMaxScaler()
    satellite_np_data = scaler.fit_transform(satellite_np_data)
    print(satellite_np_data.shape)
    index = satellite_data.index
    columns = satellite_data.columns
    time_window_size = 8
    # data_std = pd.DataFrame(satellite_np_data, index=index, columns=columns)
    # data_std.to_csv('data/data_scaler.csv', encoding='utf-8')
    # input_dataset = np.reshape(
    #         satellite_np_data,
    #         ((int)(satellite_np_data.shape[0] / time_window_size),
    #          time_window_size, satellite_np_data.shape[1]))
    ae = LstmAutoEncoder2(index, columns)

    # fit the data and save model into model_dir_path
    if DO_TRAINING:
        ae.fit(
            satellite_np_data,
            batch_size=10,
            model_dir_path=model_dir_path,
            time_window_size=time_window_size,
            estimated_negative_sample_ratio=0.9)

    # load back the model saved in model_dir_path detect anomaly
    ae.load_model(model_dir_path)
    anomaly_information = ae.anomaly(satellite_np_data)
    reconstruction_error = []
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        print('# ' + str(idx) + ' is ' +
              ('abnormal' if is_anomaly else 'normal') + ' (dist: ' +
              str(dist) + ')')
        reconstruction_error.append(dist)

    visualize_reconstruction_error(reconstruction_error, ae.threshold)


if __name__ == '__main__':
    main()
