from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, MaxPool2D
from keras.models import Sequential, Model

class models:
    def __init__(self):
        pass

    def model1(self):
        model = Sequential()
        model.add(Conv2D(256, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=(28, 28, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
        return model

    def model2(self):
        inputs = Input((28, 28, 1))
        c = Conv2D(256, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(inputs)
        c = Conv2D(256, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c)
        p = MaxPool2D((2, 2))(c)
        k = Dropout(0.2)(p)
        c = Conv2D(256, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(k)
        c = Conv2D(256, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c)
        f = Flatten()(c)
        d = Dense(512, activation="relu")(f)
        d = Dropout(0.2)(d)
        d = Dense(256, activation='relu')(d)
        d = Dropout(0.2)(d)
        d = Dense(10, activation='relu')(d)
        model = Model(inputs, d)
        model.compile(optimizer="adam",  loss="binary_crossentropy", metrics=["acc"])
        return model

