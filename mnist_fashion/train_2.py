import cv2
import os
import gzip
import numpy as np
import time
import model
from data import load_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

X_train, y_train = load_mnist(r'C:\Users\parrykhai\Desktop\mnist_fashion\data', kind='train')
X_test, y_test = load_mnist(r'C:\Users\parrykhai\Desktop\mnist_fashion\data', kind='t10k')

print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)
input_shape = (28, 28, 1)

n_classes = 10

Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

model = model.models()
model = model.model1()
#early_stopping_monitor = EarlyStopping(patience=3)

model_train = model.fit(X_train, Y_train,
                        batch_size=128,
                        epochs=3)
#print('--- %s seconds ---' % (time.time() - start))
model.save_weights("mnist.h5")

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
