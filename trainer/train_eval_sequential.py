import numpy as np
import tensorflow as tf
import keras.backend as K
import pandas as pd 

from os.path import join
from processing import load_data, normalize_inputs

from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout

SAVE_DIR = 'models/Sequential_Keras/'
MODEL_NAME = 'MNST.h5'
MODEL_DIR = join(SAVE_DIR, MODEL_NAME)

tensorboard = TensorBoard(log_dir=SAVE_DIR)


def build_sequential():

    model = Sequential([
        Dense(512, input_dim=784, kernel_initializer='normal', activation='relu'),
        Dropout(0.2),
        Dense(512, kernel_initializer='normal', activation='relu'),
        Dropout(0.2),
        Dense(10, kernel_initializer='normal', activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

train = pd.read_csv('../data/train.csv')

train_X, valid_X, train_y, valid_y = load_data(train)

train_X, train_y = normalize_inputs(train_X, train_y)
valid_X, valid_y = normalize_inputs(valid_X, valid_y)

model = build_sequential()
hx = model.fit(
    train_X,
    train_y,
    validation_data=(valid_X, valid_y),
    epochs=20,
    callbacks=[tensorboard],
    batch_size= None,
    steps_per_epoch=10,
    validation_steps=10)

#model.save(MODEL_DIR)


test = pd.read_csv('../data/test.csv')



test_X = tf.cast(test / 255, tf.float32)

test_pred = pd.DataFrame(model.predict(test_X, steps=1))
test_pred = pd.DataFrame(test_pred.idxmax(axis=1))
test_pred.index.name = 'ImageId'
test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()
test_pred['ImageId'] = test_pred['ImageId'] + 1
test_pred.to_csv('predictions/Sequential_Keras/submission.csv', index=False)


