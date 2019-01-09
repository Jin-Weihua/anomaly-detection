import numpy as np
np.random.seed(1337)  # for reproducibility
 
from keras.datasets import mnist
from keras.models import Model #泛型模型
from keras.layers import Dense, Input
from keras import metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
satellite_data1 = pd.read_csv(
    'data/data_std.csv',
    sep=',',
    index_col=0,
    encoding='utf-8',
    parse_dates=True,
    date_parser=dateparser)
# column = ['INA1_PCU输出母线电流','INA4_A电池组充电电流','INA2_A电池组放电电流','TNZ1PCU分流模块温度1','INZ6_-Y太阳电池阵电流','VNA2_A蓄电池整组电压','VNC1_蓄电池A单体1电压','VNZ2MEA电压(S3R)','VNZ4A组蓄电池BEA信号']

column = ['INA4_A电池组充电电流','INA2_A电池组放电电流','TNZ1PCU分流模块温度1','INZ6_-Y太阳电池阵电流','VNA2_A蓄电池整组电压','VNC1_蓄电池A单体1电压','VNZ2MEA电压(S3R)','VNZ4A组蓄电池BEA信号']
satellite_data = satellite_data1.loc[:,column].iloc[2920:3000]#.rolling(5).mean()#96700
satellite_data = satellite_data.dropna()
print(satellite_data.head())
satellite_np_data = satellite_data.as_matrix()
scaler = MinMaxScaler()
satellite_np_data = scaler.fit_transform(satellite_np_data)
print(satellite_np_data.shape)
index = satellite_data[0:64].index
columns = satellite_data[0:64].columns
# time_window_size = 8
# data_std = pd.DataFrame(satellite_np_data, index=index, columns=columns)
# data_std.to_csv('data/data_scaler.csv', encoding='utf-8')
# input_dataset = np.reshape(
#         satellite_np_data,
#         ((int)(satellite_np_data.shape[0] / time_window_size),
#             time_window_size, satellite_np_data.shape[1]))

x_train = satellite_np_data[0:64]
x_test = satellite_np_data[65:80]
print(x_train.shape)
print(x_test.shape)
 
# this is our input placeholder
input_data = Input(shape=(8,))
 
# 编码层
encoded = Dense(8, activation='selu')(input_data)
# encoded = Dense(64, activation='relu')(encoded)
# encoded = Dense(10, activation='relu')(encoded)
# encoder_output = Dense(encoding_dim)(encoded)
 
# 解码层
# decoded = Dense(10, activation='relu')(encoded)
# decoded = Dense(64, activation='relu')(decoded)
# decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(8, activation='selu')(encoded)
 
# 构建自编码模型
autoencoder = Model(inputs=input_data, outputs=decoded)
 
# 构建编码模型
# encoder = Model(inputs=input_data, outputs=encoder_output)
 
# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mae', metrics=[metrics.mae])
print(autoencoder.summary())
 
# training
history = autoencoder.fit(x_train, x_train,validation_data=(x_train,x_train), epochs=200, batch_size=2, shuffle=True)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
#plt.show()
plt.savefig('result/{}.png'.format('test'))

# plotting
encoded_prd = autoencoder.predict(x_train)

data_target = pd.DataFrame(x_train, index=index, columns=columns)
data_target.to_csv('data/x_train.csv', encoding='utf-8')

data_target = pd.DataFrame(encoded_prd, index=index, columns=columns)
data_target.to_csv('data/test.csv', encoding='utf-8')