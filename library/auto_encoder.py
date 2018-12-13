#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   lstm_auto_encoder.py
@Time    :   2018/12/01 20:19:35
@Author  :   靳卫华 
@Version :   1.0
@Contact :   wh.jin@hotmail.com
@Desc    :   None
'''

# here put the import lib
import matplotlib.pyplot as plt
from keras.layers import Conv1D, GlobalMaxPool1D, Dense, Flatten, LSTM, Bidirectional, RepeatVector, MaxPooling1D, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd


class LstmAutoEncoder(object):
    model_name = 'lstm-auto-encoder'
    VERBOSE = 1

    def __init__(self, index, columns):
        self.model = None
        self.time_window_size = None
        self.config = None
        self.metric = None
        self.threshold = None
        self.index = index
        self.columns = columns

    @staticmethod
    def create_model(time_window_size, metric):
        model = Sequential()
        model.add(LSTM(units=128, input_shape=(time_window_size, 1), return_sequences=False))

        model.add(Dense(units=time_window_size, activation='linear'))
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        print(model.summary())
        return model

    def load_model(self, model_dir_path):
        config_file_path = LstmAutoEncoder.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = LstmAutoEncoder.create_model(self.time_window_size, self.metric)
        weight_file_path = LstmAutoEncoder.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder.model_name + '-architecture.json'

    def fit(self, timeseries_dataset, model_dir_path, batch_size=None, epochs=None, validation_split=None, metric=None,
            estimated_negative_sample_ratio=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 20
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        self.metric = metric
        self.time_window_size = timeseries_dataset.shape[1]

        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)

        weight_file_path = LstmAutoEncoder.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = LstmAutoEncoder.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = LstmAutoEncoder.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        self.model.fit(x=input_timeseries_dataset, y=timeseries_dataset,
                       batch_size=batch_size, epochs=epochs,
                       verbose=LstmAutoEncoder.VERBOSE, validation_split=validation_split,
                       callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        scores = self.predict(timeseries_dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = LstmAutoEncoder.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        data_target = pd.DataFrame(target_timeseries_dataset, index=self.index, columns=self.columns)
        data_target.to_csv('data/data_prd.csv', encoding='utf-8')
        print(type(target_timeseries_dataset))
        dist = np.linalg.norm(timeseries_dataset - target_timeseries_dataset, axis=-1)
        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist)


class LstmAutoEncoder1(object):
    model_name = 'lstm-auto-encoder1'
    VERBOSE = 1

    def __init__(self, index, columns):
        self.model = None
        self.time_window_size = None
        self.config = None
        self.metric = None
        self.threshold = None
        self.index = index
        self.columns = columns

    @staticmethod
    def create_model(time_window_size, metric):
        model = Sequential()
        model.add(LSTM(units=9, input_shape=(time_window_size, 1), return_sequences=True))
        #model.add(LSTM(9))
        model.add(LSTM(units=time_window_size))

        #model.add(Dense(units=time_window_size, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        print(model.summary())
        return model

    def load_model(self, model_dir_path):
        config_file_path = LstmAutoEncoder1.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = LstmAutoEncoder1.create_model(self.time_window_size, self.metric)
        weight_file_path = LstmAutoEncoder1.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder1.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder1.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder1.model_name + '-architecture.json'

    def fit(self, timeseries_dataset, model_dir_path, batch_size=None, epochs=None, validation_split=None, metric=None,
            estimated_negative_sample_ratio=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 20
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        self.metric = metric
        self.time_window_size = timeseries_dataset.shape[1]

        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)

        weight_file_path = LstmAutoEncoder1.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = LstmAutoEncoder1.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = LstmAutoEncoder1.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        self.model.fit(x=input_timeseries_dataset, y=timeseries_dataset,
                       batch_size=batch_size, epochs=epochs,
                       verbose=LstmAutoEncoder1.VERBOSE, validation_split=validation_split,
                       callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        scores = self.predict(timeseries_dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = LstmAutoEncoder1.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        data_target = pd.DataFrame(target_timeseries_dataset, index=self.index, columns=self.columns)
        data_target.to_csv('data/LstmAutoEncoder1_prd.csv', encoding='utf-8')
        print(type(target_timeseries_dataset))
        dist = np.linalg.norm(timeseries_dataset - target_timeseries_dataset, axis=-1)
        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist)


class LstmAutoEncoder2(object):
    model_name = 'lstm-auto-encoder2'
    VERBOSE = 1

    def __init__(self, index, columns):
        self.model = None
        self.time_window_size = None
        self.config = None
        self.metric = None
        self.threshold = None
        self.index = index
        self.columns = columns

    @staticmethod
    def create_model(time_window_size, metric):
        model = Sequential()
        model.add(LSTM(units=18, input_shape=(time_window_size, 1), return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(9,return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(9,return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(18,return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(units=time_window_size))

        #model.add(Dense(units=time_window_size, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        print(model.summary())
        return model

    def load_model(self, model_dir_path):
        config_file_path = LstmAutoEncoder2.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = LstmAutoEncoder2.create_model(self.time_window_size, self.metric)
        weight_file_path = LstmAutoEncoder2.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder2.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder2.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder2.model_name + '-architecture.json'

    def fit(self, timeseries_dataset, model_dir_path, batch_size=None, epochs=None, validation_split=None, metric=None,
            estimated_negative_sample_ratio=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 20
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        self.metric = metric
        self.time_window_size = timeseries_dataset.shape[1]

        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)

        weight_file_path = LstmAutoEncoder2.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = LstmAutoEncoder2.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = LstmAutoEncoder2.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        self.model.fit(x=input_timeseries_dataset, y=timeseries_dataset,
                       batch_size=batch_size, epochs=epochs,
                       verbose=LstmAutoEncoder2.VERBOSE, validation_split=validation_split,
                       callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        scores = self.predict(timeseries_dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = LstmAutoEncoder2.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        data_target = pd.DataFrame(target_timeseries_dataset, index=self.index, columns=self.columns)
        data_target.to_csv('data/LstmAutoEncoder2_prd.csv', encoding='utf-8')
        print(type(target_timeseries_dataset))
        dist = np.linalg.norm(timeseries_dataset - target_timeseries_dataset, axis=-1)
        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist)




class LstmAutoEncoder3(object):
    model_name = 'lstm-auto-encoder3'
    VERBOSE = 1

    def __init__(self, index, columns):
        self.model = None
        self.time_window_size = None
        self.config = None
        self.metric = None
        self.threshold = None
        self.index = index
        self.columns = columns

    @staticmethod
    def create_model(time_window_size, metric):
        model = Sequential()
        model.add(LSTM(units=18, input_shape=(time_window_size, 1), return_sequences=True))
        # model.add(Dropout(0.4))
        model.add(LSTM(9,return_sequences=True))
        # model.add(Dropout(0.4))
        model.add(LSTM(9,return_sequences=True))
        # model.add(Dropout(0.4))
        model.add(LSTM(18,return_sequences=True))
        # model.add(Dropout(0.4))
        model.add(LSTM(units=time_window_size))

        #model.add(Dense(units=time_window_size, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        print(model.summary())
        return model

    def load_model(self, model_dir_path):
        config_file_path = LstmAutoEncoder3.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = LstmAutoEncoder3.create_model(self.time_window_size, self.metric)
        weight_file_path = LstmAutoEncoder3.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder3.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder3.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder3.model_name + '-architecture.json'

    def fit(self, timeseries_dataset, model_dir_path, batch_size=None, epochs=None, validation_split=None, metric=None,
            estimated_negative_sample_ratio=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 20
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        self.metric = metric
        self.time_window_size = timeseries_dataset.shape[1]

        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)

        weight_file_path = LstmAutoEncoder3.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = LstmAutoEncoder3.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = LstmAutoEncoder3.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        self.model.fit(x=input_timeseries_dataset, y=timeseries_dataset,
                       batch_size=batch_size, epochs=epochs,
                       verbose=LstmAutoEncoder3.VERBOSE, validation_split=validation_split,
                       callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        scores = self.predict(timeseries_dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = LstmAutoEncoder3.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        data_target = pd.DataFrame(target_timeseries_dataset, index=self.index, columns=self.columns)
        data_target.to_csv('data/LstmAutoEncoder3_prd.csv', encoding='utf-8')
        print(type(target_timeseries_dataset))
        dist = np.linalg.norm(timeseries_dataset - target_timeseries_dataset, axis=-1)
        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist)


class LstmAutoEncoder4(object):
    model_name = 'lstm-auto-encoder4'
    VERBOSE = 1

    def __init__(self, index, columns):
        self.model = None
        self.time_window_size = None
        self.config = None
        self.metric = None
        self.threshold = None
        self.index = index
        self.columns = columns

    @staticmethod
    def create_model(time_window_size, metric):
        model = Sequential()
        model.add(LSTM(units=9, input_shape=(time_window_size, 1), return_sequences=True))
        #model.add(LSTM(9))
        model.add(LSTM(units=time_window_size))

        #model.add(Dense(units=time_window_size, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        print(model.summary())
        return model

    def load_model(self, model_dir_path):
        config_file_path = LstmAutoEncoder4.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = LstmAutoEncoder4.create_model(self.time_window_size, self.metric)
        weight_file_path = LstmAutoEncoder4.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder4.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder4.model_name + '-weights.{epoch:02d}-{val_loss:.8f}.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder4.model_name + '-architecture.json'

    def fit(self, timeseries_dataset, model_dir_path, batch_size=None, epochs=None, validation_split=None, metric=None,
            estimated_negative_sample_ratio=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 30
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        self.metric = metric
        self.time_window_size = timeseries_dataset.shape[1]

        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)

        weight_file_path = LstmAutoEncoder4.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = LstmAutoEncoder4.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = LstmAutoEncoder4.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        history = self.model.fit(x=input_timeseries_dataset, y=timeseries_dataset,
                       batch_size=batch_size, epochs=epochs,
                       verbose=LstmAutoEncoder4.VERBOSE, validation_split=validation_split,
                       callbacks=[checkpoint])
        #self.model.save_weights(weight_file_path)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train","test"],loc="upper left")
        #plt.show()
        plt.savefig('result/{}.png'.format(self.model_name))

        scores = self.predict(timeseries_dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = LstmAutoEncoder4.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        data_target = pd.DataFrame(target_timeseries_dataset, index=self.index, columns=self.columns)
        data_target.to_csv('data/LstmAutoEncoder4_prd.csv', encoding='utf-8')
        print(type(target_timeseries_dataset))
        dist = np.linalg.norm(timeseries_dataset - target_timeseries_dataset, axis=-1)
        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist)

class CnnLstmAutoEncoder(object):
    model_name = 'cnn-lstm-auto-encoder'
    VERBOSE = 1

    def __init__(self):
        self.model = None
        self.time_window_size = None
        self.config = None
        self.metric = None
        self.threshold = None

    @staticmethod
    def create_model(time_window_size, metric):
        model = Sequential()

        model.add(Conv1D(filters=256, kernel_size=9, padding='same', activation='relu',
                         input_shape=(time_window_size, 1)))
        model.add(MaxPooling1D(pool_size=4))

        model.add(LSTM(64))

        model.add(Dense(units=time_window_size, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])

        # model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        # model.compile(optimizer="sgd", loss="mse", metrics=[metric])

        print(model.summary())
        return model

    def load_model(self, model_dir_path):
        config_file_path = CnnLstmAutoEncoder.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = CnnLstmAutoEncoder.create_model(self.time_window_size, self.metric)
        weight_file_path = CnnLstmAutoEncoder.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + CnnLstmAutoEncoder.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + CnnLstmAutoEncoder.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + CnnLstmAutoEncoder.model_name + '-architecture.json'

    def fit(self, timeseries_dataset, model_dir_path, batch_size=None, epochs=None, validation_split=None, metric=None,
            estimated_negative_sample_ratio=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 20
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        self.metric = metric
        self.time_window_size = timeseries_dataset.shape[1]

        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)

        weight_file_path = CnnLstmAutoEncoder.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = CnnLstmAutoEncoder.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = CnnLstmAutoEncoder.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        self.model.fit(x=input_timeseries_dataset, y=timeseries_dataset,
                       batch_size=batch_size, epochs=epochs,
                       verbose=CnnLstmAutoEncoder.VERBOSE, validation_split=validation_split,
                       callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        scores = self.predict(timeseries_dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = CnnLstmAutoEncoder.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        dist = np.linalg.norm(timeseries_dataset - target_timeseries_dataset, axis=-1)
        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist)


class BidirectionalLstmAutoEncoder(object):
    model_name = 'bidirectional-lstm-auto-encoder'
    VERBOSE = 1

    def __init__(self):
        self.model = None
        self.time_window_size = None
        self.config = None
        self.metric = None
        self.threshold = None

    @staticmethod
    def create_model(time_window_size, metric):
        model = Sequential()

        model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2), input_shape=(time_window_size, 1)))

        model.add(Dense(units=time_window_size, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])

        # model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        # model.compile(optimizer="sgd", loss="mse", metrics=[metric])

        print(model.summary())
        return model

    def load_model(self, model_dir_path):
        config_file_path = BidirectionalLstmAutoEncoder.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = BidirectionalLstmAutoEncoder.create_model(self.time_window_size, self.metric)
        weight_file_path = BidirectionalLstmAutoEncoder.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + BidirectionalLstmAutoEncoder.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + BidirectionalLstmAutoEncoder.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + BidirectionalLstmAutoEncoder.model_name + '-architecture.json'

    def fit(self, timeseries_dataset, model_dir_path, batch_size=None, epochs=None, validation_split=None, metric=None,
            estimated_negative_sample_ratio=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 20
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        self.metric = metric
        self.time_window_size = timeseries_dataset.shape[1]

        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)

        weight_file_path = BidirectionalLstmAutoEncoder.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = BidirectionalLstmAutoEncoder.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = BidirectionalLstmAutoEncoder.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        self.model.fit(x=input_timeseries_dataset, y=timeseries_dataset,
                       batch_size=batch_size, epochs=epochs,
                       verbose=BidirectionalLstmAutoEncoder.VERBOSE, validation_split=validation_split,
                       callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        scores = self.predict(timeseries_dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = BidirectionalLstmAutoEncoder.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        dist = np.linalg.norm(timeseries_dataset - target_timeseries_dataset, axis=-1)
        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist)