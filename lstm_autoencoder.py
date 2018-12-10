#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   lstm_autoencoder.py
@Time    :   2018/12/06 14:27:42
@Author  :   靳卫华 
@Version :   1.0
@Contact :   wh.jin@hotmail.com
@Desc    :   None
'''

# here put the import lib

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from library.plot_utils import visualize_reconstruction_error
from library.auto_encoder import LstmAutoEncoder

DO_TRAINING = False


def main():
    data_dir_path = 'data'
    model_dir_path = 'model'
    dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    satellite_data = pd.read_csv(
        data_dir_path + '/1208_new.csv',
        sep=',',
        index_col=0,
        encoding='utf-8',
        parse_dates=True,
        date_parser=dateparser)
    print(satellite_data.head())
    satellite_np_data = satellite_data.as_matrix()
    scaler = MinMaxScaler()
    satellite_np_data = scaler.fit_transform(satellite_np_data)
    print(satellite_np_data.shape)
    index = satellite_data.index
    columns = satellite_data.columns
    # data_std = pd.DataFrame(satellite_np_data, index=index, columns=columns)
    # data_std.to_csv('data/data_std_train.csv', encoding='utf-8')

    ae = LstmAutoEncoder()

    # fit the data and save model into model_dir_path
    if DO_TRAINING:
        ae.fit(
            satellite_np_data[:96717, :],
            model_dir_path=model_dir_path,
            estimated_negative_sample_ratio=0.9)

    # load back the model saved in model_dir_path detect anomaly
    ae.load_model(model_dir_path)
    anomaly_information = ae.anomaly(satellite_np_data[:96717, :], index, columns)
    reconstruction_error = []
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        print('# ' + str(idx) + ' is ' +
              ('abnormal' if is_anomaly else 'normal') + ' (dist: ' +
              str(dist) + ')')
        reconstruction_error.append(dist)

    visualize_reconstruction_error(reconstruction_error, ae.threshold)


if __name__ == '__main__':
    main()
