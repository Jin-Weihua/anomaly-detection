# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 14:11:01 2018

@author: admin
"""

import keras
import keras.backend as K
import pickle
from keras.layers.core import Activation
from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Dropout, LSTM,CuDNNLSTM
from numpy import nan as NaN
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing


# load the data
df = pd.read_csv("data/jsdata_with.csv")
df2 = df.iloc[:,0:4]
overview = df2.describe()

# EDA %matplotlib qt5
def df_plot(df,length=None):
# %matplotlib qt5
    fig = plt.figure()
    num_feat = df.shape[1]
    columns = df.columns.tolist()
    for i in range(num_feat):
        if length==None:
            plt.plot(df.iloc[:,i],label=str(columns[i]))
        else:
            plt.plot(df.iloc[:length,i],label=str(columns[i]))
    plt.xlabel("Time_step",fontsize=25)
    plt.show()
    
def df_subplot(df,length=None):
# %matplotlib qt5
    fig = plt.figure()
    num_feat = df.shape[1]
    columns = df.columns.tolist()
    
    for i in range(num_feat):
        axTemp = fig.add_subplot(num_feat,1,i+1)
        if length==None:
            axTemp.plot(df.iloc[:,i],label=str(columns[i]))
        else:
            axTemp.plot(df.iloc[:length,i],label=str(columns[i]))
    plt.xlabel("Time_step",fontsize=25)
    plt.show()

df_subplot(df2,2000)
df_subplot(df2)

# 生成标签列Y
y_label = np.ones(df2.shape[0])
df2['Y'] = y_label
df2['Y'][df2['currentI']>1500] =0
Y_training = df2['Y']
# 数据集处理 training/testing 生成
split_rate = 0.8
length = df2.shape[0]
df_training = df2[0:int(length*split_rate)]
df_testing = df2[int(length*split_rate):]

# 训练集归一化，生成归一化器 minmaxscaler
min_max_scaler = preprocessing.MinMaxScaler()   # 
cols_normalize = df_training.columns.difference(['Y'])  #对标签以外的数据做归一化
min_max_scaler.fit(df_training[cols_normalize])
norm_train_df = pd.DataFrame(min_max_scaler.transform(df_training[cols_normalize]), 
                                 columns=cols_normalize, 
                                 index=df_training.index)

# save min_max_scaler
pickle.dump(min_max_scaler,open("D:\\lzd\\lstmClassifier\\minmaxscaler.p","wb"))

df_training_norm = norm_train_df
df_subplot(df_training_norm,2000)

df_training_with_Y = df_training_norm.copy()
df_training_with_Y['Y'] = df2['Y']
# 训练集切片 batch
window_size = 20
featureSize = norm_train_df.shape[1]

#生成训练网络用的batch
def gen_sequence(id_df, seq_length, seq_cols):    # 对id_df[seq_cols]做切片，切片长度seq_length, 

    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

# 训练集切成batch
seq_gen = list(gen_sequence(df_training_with_Y, window_size, list(df_training_with_Y.columns)))
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)    
batch_training_temp = seq_array
print (batch_training_temp.shape)
num = batch_training_temp.shape[0]

X_training_batch = batch_training_temp[:,0:4].copy()
Y_training_batch = batch_training_temp[:,[4]].copy()

X_training_batch = X_training_batch.reshape(int(num/window_size),window_size,featureSize)
Y_training_batch = Y_training_batch.reshape(int(num/window_size),window_size)

num_sample = Y_training_batch.shape[0]
temp = np.ones(num_sample)
temp.reshape(num_sample,1)
for i in range(num_sample):
    if Y_training_batch[i,:].min() == 0:
        temp[i]=0

Y_training_batch = temp    




# 建立lstm网络 二分类
# parameters
epo = 1000
bat_size = 250
val_split = 0.2

first_units = 9
second_units = 6

model_path = "D:\\lzd\\lstmClassifier\\lstmAutoEncoder2_tanh_mae_3units.hdf5"

model = Sequential()
model.add(CuDNNLSTM(
         input_shape=(window_size, featureSize),
         units=3,
         return_sequences=True,name="LSTM1"))
#model.add(Dropout(0.1))
#model.add(CuDNNLSTM(
#          units=second_units,
#          return_sequences=True,name="LSTM2"))
#model.add(CuDNNLSTM(
#          units=featureSize,
#          return_sequences=True,name="LSTM2"))
#model.add(Dropout(0.1))
model.add(Dense(featureSize))
model.add(Activation('tanh')) #softmax
model.compile(loss='mae', # multi_crossentropy,mean_squared_error
              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
              metrics=['mean_absolute_error'])

print (model.summary())


#model.compile(loss='mae', optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08))


history = model.fit(X_training_batch, X_training_batch, epochs=epo, batch_size=bat_size, validation_split=val_split, verbose=2,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
          )
#model.compile(loss='mae', optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
print (model.summary())
#model_trained = load_model("D:\\lzd\\lstmClassifier\\lstmClassifier.hdf5")
#model_trained.save("D:\\lzd\\lstmClassifier\\lstmClassifier.hdf5")
model_trained2 = load_model(model_path)

min_max_scaler = pickle.load( open( "D:\\lzd\\lstmClassifier\\minmaxscaler.p", "rb" ) )

#对测试集归一化

df_testing_norm = pd.DataFrame(min_max_scaler.transform(df_testing[cols_normalize]), 
                                 columns=cols_normalize, 
                                 index=df_testing.index)

df_testing_with = df_testing_norm.copy()
#df_testing_with_Y['Y'] = df2['Y'][int(length*split_rate):]

df_subplot(df_testing_with,4000)

#seq_gen = list(gen_sequence(df_testing_with, window_size, list(df_testing_with.columns)))
#seq_array = np.concatenate(list(seq_gen)).astype(np.float32)    
#batch_testing_temp = seq_array
#print (batch_testing_temp.shape)
#num = batch_testing_temp.shape[0]

df_X_testing_show_fault = df_testing_with.copy()
df_X_testing_show_fault.iloc[:,0][100:200]=0.1

X_testing_batch_array = df_X_testing_show_fault[0:4000].values
X_testing_batch_array = X_testing_batch_array.reshape(int(4000/window_size),window_size,X_testing_batch_array.shape[1])
print ('X_testing_batch_array: '+str(X_testing_batch_array.shape))

#X_testing_batch = batch_testing_temp[:,0:4].copy()
#Y_testing_batch = batch_testing_temp[:,[4]].copy()

#X_testing_batch = X_testing_batch.reshape(int(num/20),window_size,featureSize)
#Y_testing_batch = Y_testing_batch.reshape(int(num/20),window_size)

#num_sample = X_testing_batch.shape[0]
#temp = np.ones(num_sample)
#temp.reshape(num_sample,1)
#for i in range(num_sample):
#    if Y_testing_batch[i,:].min() == 0:
#        temp[i]=0
#
#Y_testing_batch = temp    


#scores = model_trained2.evaluate(X_testing_batch, Y_testing_batch, verbose=1, batch_size=200)
result_prediction = model_trained2.predict(X_testing_batch_array)
#result2 = model_trained2.predict_classes(X_testing_batch)
print ('result_prediction: '+str(result_prediction.shape))

#df_testing_show = df_testing_with_Y[window_size:].copy()
#df_testing_show['result']=result
#df_testing_show['result2']=result2

X_testing_show = X_testing_batch_array.reshape(X_testing_batch_array.shape[0]*X_testing_batch_array.shape[1],X_testing_batch_array.shape[2])
Y_testing_show = result_prediction.reshape(result_prediction.shape[0]*result_prediction.shape[1],result_prediction.shape[2])

df_X_testing_show = pd.DataFrame(X_testing_show)
df_Y_testing_show = pd.DataFrame(Y_testing_show)

df_subplot(df_X_testing_show,4000)
df_subplot(df_Y_testing_show,4000)

# 故障注入




# 输出中间结果
#model_temp = load_model("D:\\lzd\\lstmClassifier\\lstmClassifier2.hdf5")
LSTM1_model = Model(inputs=model_temp.input,outputs=model_temp.get_layer('LSTM1').output)
LSTM2_model = Model(inputs=model_temp.input,outputs=model_temp.get_layer('LSTM2').output)
LSTM1_output = LSTM1_model.predict(X_testing_batch)
LSTM2_output = LSTM2_model.predict(X_testing_batch)

num_test_sample = LSTM1_output.shape[0]
LSTM1_output_show = np.ones([num_test_sample,LSTM1_output.shape[2]])
LSTM2_output_show = np.ones([num_test_sample,LSTM2_output.shape[1]])
for i in range(num_test_sample):
    LSTM1_output_show[i:]=LSTM1_output[i,window_size-1,:]
    LSTM2_output_show[i:] = LSTM2_output[i,:]



df_LSTM1_output_show = pd.DataFrame(LSTM1_output_show)
df_LSTM2_output_show = pd.DataFrame(LSTM2_output_show)

df_subplot(df_testing_show,2000)
df_testing_show_original = df_testing_show.drop(['result','result2','Y'],axis=1)
df_subplot(df_testing_show_original,2000)
df_subplot(df_LSTM1_output_show,2000)
df_subplot(df_LSTM2_output_show,2000)

