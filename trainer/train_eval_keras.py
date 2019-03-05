import numpy as np
import tensorflow as tf
import keras.backend as K
import pandas as pd 

from processing import load_data
from keras.models import Sequential
from keras.layers import Dense

def normalize_inputs(x, y):
    x = tf.cast(x / 255, tf.float32)
    y = tf.one_hot(indices=[i for i in range(0, 10)], depth=10)
    return x, y


def build_sequential():

    model = Sequential([
        Dense(800, input_dim=784, kernel_initializer='normal'),
        Dense(10, kernel_initializer='normal', activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

train = pd.read_csv('../data/train.csv')

train_X, valid_X, train_y, valid_y = load_data(train)

train_X, train_y = normalize_inputs(train_X, train_y)

print(train_X)
print(train_y)

#model = build_sequential()

#model.fit(train_X, train_y, validation_data=(valid_X, valid_y), epochs=10, bat)




