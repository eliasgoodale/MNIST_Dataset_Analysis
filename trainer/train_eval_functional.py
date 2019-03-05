import pandas as pd
import numpy as np

from keras.callbacks import TensorBoard
from keras.layers import Input, Dense
from keras.models import Model
from processing import load_data, normalize_inputs

SAVE_DIR = 'models/Functional_Keras/'
tensorboard = TensorBoard(log_dir=SAVE_DIR)

train = pd.read_csv('../data/train.csv')
test_X = pd.read_csv('../data/test.csv')

train_X, valid_X, train_y, valid_y = load_data(train)


train_X, train_y = normalize_inputs(train_X, train_y)
valid_X, valid_y = normalize_inputs(valid_X, valid_y)


inputs = Input(shape=(784,))

h1 = Dense(512, activation='relu')(inputs)
h2 = Dense(512, activation='relu')(h1)

pred = Dense(10, activation='softmax')(h2)

model = Model(inputs=inputs, outputs=pred)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hx = model.fit(
    train_X,
    train_y,
    validation_data=(valid_X, valid_y),
    epochs=20,
    callbacks=[tensorboard],
    batch_size= None,
    steps_per_epoch=10,
    validation_steps=10)

test_pred = pd.DataFrame(model.predict(test_X, steps=1))
test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))
test_pred.index.name = 'ImageId'
test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()
test_pred['ImageId'] = test_pred['ImageId'] + 1

test_pred.to_csv('predictions/Functional_Keras/submission.csv', index = False)