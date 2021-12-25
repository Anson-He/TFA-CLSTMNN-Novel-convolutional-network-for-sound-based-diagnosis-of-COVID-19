import os
import pickle
import librosa
import numpy as np
from time import *


# 加载音频
def wav_load(file):
    wav_data, sr = librosa.load(file, duration=10)
    wav_data = wav_data.tolist()
    while len(wav_data) < 220500:
        wav_data.append(0)  # 长度归一化
    wav_data = np.array(wav_data)
    return wav_data, sr


# MFCC特征提取
def get_matrix(wav_data, sr):
    return np.mean(librosa.feature.mfcc(wav_data, sr=sr, n_mfcc=40), axis=0)
# def get_matrix(wav_data, sr):
#     return librosa.feature.mfcc(wav_data, sr=sr, n_mfcc=40).T


data = []
labels = []
Positive = os.listdir('./data/tmp')
Negative = os.listdir('./data/tmp2')

begin = time()

for p in Positive:
    wav, s = wav_load('./data/tmp/'+p)
    matrix = get_matrix(wav, s)
    data.append(matrix)
    labels.append(1)
for n in Negative:
    wav, s = wav_load('./data/tmp2/' + n)
    matrix = get_matrix(wav, s)
    data.append(matrix)
    labels.append(0)

end = time()
data = np.array(data)
labels = np.array(labels)
print('done 耗时', end-begin)

# 划分数据集
ratioTrain = 0.8
numTrain = int(data.shape[0] * ratioTrain)
permutation = np.random.permutation(data.shape[0])
data = data[permutation, :]
labels = labels[permutation]

x_train = data[:numTrain]
x2 = data[numTrain:]
y_train = labels[:numTrain]
y2 = labels[numTrain:]

ratioVal = 0.8
numVal = int(x2.shape[0] * ratioVal)
permutation2 = np.random.permutation(x2.shape[0])
x2 = x2[permutation2, :]
y2 = y2[permutation2]

x_val = x2[:numVal]
x_test = x2[numVal:]
y_val = y2[:numVal]
y_test = y2[numVal:]

Val = os.listdir('./data/val')
Test = os.listdir('./data/test')

x_val = x_val.tolist()
y_val = y_val.tolist()
x_test = x_test.tolist()
y_test = y_test.tolist()
for val in Val:
    wav, s = wav_load('./data/val/' + val)
    matrix = get_matrix(wav, s)
    x_val.append(matrix)
    y_val.append(1)
for test in Test:
    wav, s = wav_load('./data/test/' + test)
    matrix = get_matrix(wav, s)
    x_test.append(matrix)
    y_test.append(1)

x_val = np.array(x_val)
y_val = np.array(y_val)
x_test = np.array(x_test)
y_test = np.array(y_test)

print('训练样本尺寸为：', x_train.shape)
print('训练标签尺寸为：', y_train.shape)
print('验证样本尺寸为：', x_val.shape)
print('验证标签尺寸为：', y_val.shape)
print('测试样本尺寸为：', x_test.shape)
print('测试标签尺寸为：', y_test.shape)


dic = dict()
dic['x_train'] = x_train
dic['x_val'] = x_val
dic['x_test'] = x_test
dic['y_train'] = y_train
dic['y_val'] = y_val
dic['y_test'] = y_test

with open('mfcc_model_para_dict.pkl', 'wb') as f:
    pickle.dump(dic, f)
f.close()
print('done')

# with open('mfcc_LSTM.pkl', 'wb') as f:
#     pickle.dump(dic, f)
# f.close()
# print('done')
