import pickle
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
import tensorflow
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import multiply, Permute
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.utils.vis_utils import plot_model


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


with open('mfcc_LSTM.pkl', 'rb') as f:
    paradict = pickle.load(f)


x_train = paradict['x_train']
x_train = np.array(x_train)
# x_train = normalization(x_train)
# x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
# DATA_MEAN = np.mean(x_train, axis=0)
# DATA_STD = np.std(x_train, axis=0)
# x_train -= DATA_MEAN
# x_train /= DATA_STD

x_val = paradict['x_val']
x_val = np.array(x_val)
# x_val = normalization(x_val)
# x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
# DATA_MEAN = np.mean(x_val, axis=0)
# DATA_STD = np.std(x_val, axis=0)
# x_val -= DATA_MEAN
# x_val /= DATA_STD

x_test = paradict['x_test']
x_test = np.array(x_test)
# x_test = normalization(x_test)
# x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
# DATA_MEAN = np.mean(x_test, axis=0)
# DATA_STD = np.std(x_test, axis=0)
# x_test -= DATA_MEAN
# x_test /= DATA_STD

y_train = paradict['y_train']
y_train = np.array(y_train)
y_train = to_categorical(y_train)
y_val = paradict['y_val']
y_val = np.array(y_val)
y_val = to_categorical(y_val)
y_test = paradict['y_test']
y_test = np.array(y_test)
y_test = to_categorical(y_test)
print('训练样本尺寸为：', x_train.shape)
print('训练标签尺寸为：', y_train.shape)
print('验证样本尺寸为：', x_val.shape)
print('验证标签尺寸为：', y_val.shape)
print('测试样本尺寸为：', x_test.shape)
print('测试标签尺寸为：', y_test.shape)


inputs = Input(shape=(431, 40))
BN = layers.BatchNormalization()(inputs)
Con1 = layers.Conv1D(128, 7, activation='relu', kernel_regularizer=regularizers.l2(0.01))(BN)
Drop1 = layers.Dropout(0.2)(Con1)
Pool1 = layers.MaxPooling1D(pool_size=(7))(Drop1)
Con2 = layers.Conv1D(128, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01))(Pool1)
Drop2 = layers.Dropout(0.2)(Con2)
Pool2 = layers.MaxPooling1D(pool_size=(5))(Drop2)
# 频域注意力
a = layers.Dense(Pool2.shape[2], activation='softmax')(Pool2)
a_out = multiply([Pool2, a])

D1 = layers.Dense(32)(a_out)
Drop3 = layers.Dropout(0.2)(D1)

BN2 = layers.BatchNormalization()(Drop3)
Lstm1 = layers.LSTM(128, activation='tanh', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(BN2)
Drop4 = layers.Dropout(0.2)(Lstm1)
Lstm2 = layers.LSTM(128, activation='tanh', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(Drop4)
Drop5 = layers.Dropout(0.2)(Lstm2)
# 时域注意力
a2 = Permute((2, 1))(Drop5)
a2 = layers.Dense(Drop5.shape[1], activation='softmax')(a2)
a2_probs = Permute((2, 1))(a2)
a2_out = multiply([Drop5, a2_probs])

flatten = layers.Flatten()(a2_out)
D2 = layers.Dense(32)(flatten)
Drop6 = layers.Dropout(0.2)(D2)
D3 = layers.Dense(32)(Drop6)
Drop7 = layers.Dropout(0.2)(D3)
output = layers.Dense(2, activation='softmax')(Drop7)
model = Model(inputs=inputs, outputs=output)


# model = models.Sequential()
# model.add(layers.BatchNormalization(input_shape=(302, 40)))
# model.add(layers.Conv1D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# model.add(layers.Dropout(0.2))
# model.add(layers.MaxPooling1D(pool_size=(3)))
# model.add(layers.Conv1D(32, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# model.add(layers.Dropout(0.2))
# model.add(layers.MaxPooling1D(pool_size=(3)))
# model.add(layers.Dense(256))
# model.add(layers.Dropout(0.2))
# model.add(layers.LSTM(64, activation='tanh', kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
# model.add(layers.Dropout(0.2))
# model.add(layers.LSTM(32, activation='tanh', kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
# model.add(layers.Dropout(0.2))
# model.add(layers.Flatten())
# model.add(layers.Dense(256))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(256))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(2, activation='softmax'))

plot_model(model, to_file='./TFA-CLSTMNN/TFA-CLSTMNN_model.png', show_shapes=True)
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

callbacks_list=[
    tensorflow.keras.callbacks.EarlyStopping(
        monitor='accuracy',
        patience=10,
    ),
    tensorflow.keras.callbacks.ModelCheckpoint(
        filepath='./TFA-CLSTMNN/TFA-CLSTMNN_checkpoint.h5',
        monitor='val_loss',
        save_best_only=True
    ),
    tensorflow.keras.callbacks.TensorBoard(
        log_dir='./TFA-CLSTMNN/TFA-CLDNN_train_log'
    )
]
history=model.fit(x_train, y_train,
                  batch_size=16,
                  epochs=100,
                  validation_data=(x_val, y_val),
                 callbacks=callbacks_list)
model.save('./TFA-CLDNN/TFA-CLDNN_model.h5')
model.save_weights('./TFA-CLDNN/TFA-CLSTMNN_weight.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print(model.evaluate(x_test, y_test))