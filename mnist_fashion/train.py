import os
import gzip
import numpy as np
import time


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28, 28)

    # for flat arrays:
    # with gzip.open(images_path, 'rb') as imgpath:
    #     images = np.frombuffer(imgpath.read(), dtype=np.uint8,
    #                            offset=16).reshape(len(labels), 784)

    return images, labels


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

model = Sequential()
model.add(Conv2D(256, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_classes, activation='softmax'))
start = time.time()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early_stopping_monitor = EarlyStopping(patience=3)

model_train = model.fit(X_train, Y_train,
                        batch_size=128,
                        epochs=3,
                        verbose=0,
                        callbacks=[early_stopping_monitor],
                        validation_data=(X_test, Y_test))
print('--- %s seconds ---' % (time.time() - start))
model.save_weights("mnist.h5")

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
