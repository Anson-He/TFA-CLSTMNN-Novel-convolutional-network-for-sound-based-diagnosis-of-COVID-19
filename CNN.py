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
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.utils.vis_utils import plot_model

with open('mfcc_model_para_dict2.pkl', 'rb') as f:
    paradict = pickle.load(f)
x_train = paradict['x_train']
x_train = np.array(x_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
DATA_MIN = np.min(x_train, axis=0)
DATA_MAX = np.max(x_train, axis=0)
x_train = x_train - DATA_MIN
x_train = x_train/(DATA_MAX-DATA_MIN)
# DATA_MEAN = np.mean(x_train, axis=0)
# DATA_STD = np.std(x_train, axis=0)
# x_train -= DATA_MEAN
# x_train /= DATA_STD

x_val = paradict['x_val']
x_val = np.array(x_val)
x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
DATA_MIN = np.min(x_val, axis=0)
DATA_MAX = np.max(x_val, axis=0)
x_val = x_val - DATA_MIN
x_val = x_val/(DATA_MAX-DATA_MIN)
# DATA_MEAN = np.mean(x_val, axis=0)
# DATA_STD = np.std(x_val, axis=0)
# x_val -= DATA_MEAN
# x_val /= DATA_STD

x_test = paradict['x_test']
x_test = np.array(x_test)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
DATA_MIN = np.min(x_test, axis=0)
DATA_MAX = np.max(x_test, axis=0)
x_test = x_test - DATA_MIN
x_test = x_test/(DATA_MAX-DATA_MIN)
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

model = models.Sequential()
model.add(layers.BatchNormalization(input_shape=(431, 1)))
model.add(layers.Conv1D(16, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling1D(pool_size=(3)))
model.add(layers.Conv1D(16, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling1D(pool_size=(3)))
model.add(layers.Flatten())
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(2, activation='softmax'))

plot_model(model, to_file='./CNN/CNN.png', show_shapes=True)
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

callbacks_list=[
    tensorflow.keras.callbacks.EarlyStopping(
        monitor='accuracy',
        patience=10,
    ),
    tensorflow.keras.callbacks.ModelCheckpoint(
        filepath='./CNN/CNN_checkpoint.h5',
        monitor='val_loss',
        save_best_only=True
    ),
    tensorflow.keras.callbacks.TensorBoard(
        log_dir='./CNN/CNN_train_log'
    )
]
history=model.fit(x_train, y_train,
                  batch_size=16,
                  epochs=100,
                  validation_data=(x_val, y_val),
                 callbacks=callbacks_list)
model.save('./CNN/CNN.h5')
model.save_weights('./CNN/CNN_weight.h5')

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