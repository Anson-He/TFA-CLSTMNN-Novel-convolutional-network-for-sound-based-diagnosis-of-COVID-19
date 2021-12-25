import os
import pickle
import librosa
import numpy as np
from time import *
from tensorflow.keras.models import load_model

# 加载模型
model = load_model('./TFA-CLDNN/TFA-CLDNN_model.h5')
Positive = os.listdir('./data/test')
Negative = os.listdir('./data/Negative')
res = []
for i in Positive:
    wav_data, sr = librosa.load('./data/Positive/'+i, duration=10)
    wav_data = wav_data.tolist()
    while len(wav_data) < 220500:
        wav_data.append(0)  # 长度归一化
    wav_data = np.array(wav_data)
    x = librosa.feature.mfcc(wav_data, sr=sr, n_mfcc=40).T
    x = x.reshape(1, 431, 40)
    # x = np.mean(librosa.feature.mfcc(wav_data, sr=sr, n_mfcc=40), axis=0)
    # x = x.reshape(1, 431, 1)
    res.append(list(model.predict(x)[0]).index(max(list(model.predict(x)[0]))))
res2 = []
for i in Negative:
    wav_data, sr = librosa.load('./data/Negative/'+i, duration=10)
    wav_data = wav_data.tolist()
    while len(wav_data) < 220500:
        wav_data.append(0)  # 长度归一化
    wav_data = np.array(wav_data)
    x = librosa.feature.mfcc(wav_data, sr=sr, n_mfcc=40).T
    x = x.reshape(1, 431, 40)
    # x = np.mean(librosa.feature.mfcc(wav_data, sr=sr, n_mfcc=40), axis=0)
    # x = x.reshape(1, 431, 1)
    res2.append(list(model.predict(x)[0]).index(max(list(model.predict(x)[0]))))
s = 0
for j in res:
    if j == 1:
        s = s + 1
print('正样本召回率为:', s/len(res))
s2 = 0
for j in res2:
    if j == 1:
        s2 = s2 + 1
print('反样本误判率为:', s2/len(res2))
# # 加载音频
# def wav_load(file):
#     wav_data, sr = librosa.load(file, duration=3)
#     wav_data = wav_data.tolist()
#     while len(wav_data) < 154350:
#         wav_data.append(0)  # 长度归一化
#     wav_data = np.array(wav_data)
#     return wav_data, sr
#
#
# # MFCC特征提取
# # def get_matrix(wav_data, sr):
# #     return np.mean(librosa.feature.mfcc(wav_data, sr=sr, n_mfcc=40), axis=0)
# def get_matrix(wav_data, sr):
#     return librosa.feature.mfcc(wav_data, sr=sr, n_mfcc=40).T
#
#
# data = []
# labels = []
# Positive = os.listdir('./data/Positive')
# Negative = os.listdir('./data/Negative')
#
# begin = time()
#
# for p in Positive:
#     wav, s = wav_load('./data/Positive/'+p)
#     matrix = get_matrix(wav, s)
#     data.append(matrix)
#     labels.append(1)
# for n in Negative:
#     wav, s = wav_load('./data/Negative/' + n)
#     matrix = get_matrix(wav, s)
#     data.append(matrix)
#     labels.append(0)
#
# end = time()
# data = np.array(data)
# labels = np.array(labels)
# print('done 耗时', end-begin)


