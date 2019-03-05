import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(df):
    train_y = df['label']
    train_X = df.drop('label', axis=1)

    return train_test_split(train_X, train_y, test_size=0.3, random_state=0)

def preprocess_data(x, y):
    labels = tf.cast(y, tf.int32)
    input_data = tf.cast(x, tf.float32)
    return (dict({'image': input_data}), labels)

def input_fn(train_X, train_y, batch_size=64, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))

    dataset = dataset.map(lambda x,y: preprocess_data(x, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=128)
    
    dataset = dataset.batch(batch_size)

    itr = dataset.make_one_shot_iterator()
    features, target = itr.get_next()

    return features, target