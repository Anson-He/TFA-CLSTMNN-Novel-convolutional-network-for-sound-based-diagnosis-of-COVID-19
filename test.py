import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import numpy as np
wav_data, sr = librosa.load('./data/Positive/1133_Positive_male_49_cough.wav', duration=3)
wav_data = wav_data.tolist()
while len(wav_data) < 66150:
    wav_data.append(0)  # 长度归一化
wav_data = np.array(wav_data)
# melspec = librosa.feature.melspectrogram(wav_data, sr, n_fft=1024, hop_length=512, n_mels=128)
mfcc = librosa.feature.mfcc(wav_data, sr, n_mfcc=40)
print(mfcc.T)
plt.matshow(mfcc)
plt.title('MFCC')
plt.show()
