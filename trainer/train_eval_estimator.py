import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

'''
In this task we define 3 types of models.

We have 784 inputs and 10 outputs so the network size is calculated first by the mean of input/output

(784 + 10) /2 = 397 round to 400 hidden layer neurons

Wiki page claims that the a low error rate(1.6) for a deep nn is acheived with a
784-800-10 topology and no elastic distortion

TOPOLOGY = 784-800-10
LEARNING_RATE = 0.0001

Optimizer: Case: Adam
    The Adam algorithm computes individual adaptive learning rates for 
    different parameters from estimates of first and second moments of the gradients.

    It combines: 
        >   Adaptive Gradient Alogrithm(AdaGrad) that maintains a per-parameter learning
        rate that improve performance on problems with sparse gradients( GOOD FOR COMPUTER VISION)
        >   Root Mean Square Propagation(RMSProp) that maintains a per-parameter learning
        rate that is adapted based on the recent magnitudes of the gradients for the weight
    
Non-stationary data, as a rule, are unpredictable and cannot be modeled or forecasted. 
The results obtained by using non-stationary time series may be spurious in that they may 
indicate a relationship between two variables where one does not exist.

Since we have sparse 1 hot vectors for each one the pixels, and none of the pixels
are directly related to another through some continuous fashion it is a good idea to use Adam

The output layer is activated with softmax



1.) DNNRegressor(
    hidden_units,
    feature_columns,
    model_dir=None,
    n_classes=2,
    weight_column=None,
    label_vocabulary=None,
    optimizer='Adagrad',
    activation_fn=tf.nn.relu,
    dropout=None,
    input_layer_partitioner=None,
    config=None,
    warm_start_from=None,
    loss_reduction=losses.Reduction.SUM,
    batch_norm=False
)
'''

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


def build_DNNClassifier():
    checkpoint_config = tf.estimator.RunConfig(
        save_checkpoints_secs = 5 * 60,
        keep_checkpoint_max = 10
    )
    feature_columns = [tf.feature_column.numeric_column(key="image", shape=(784, ))]

    # Build single layer of hidden units 800-10

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[800],
        optimizer=tf.train.AdamOptimizer(1e-4),
        n_classes=10,
        dropout=0.1,
        model_dir='models/TensorflowDNNClassifier',
        config=checkpoint_config
    )
    return classifier

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')


train = pd.read_csv('../data/train.csv')

train_X, valid_X, train_y, valid_y = load_data(train)

estimator = build_DNNClassifier()

train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: input_fn(train_X, train_y, 64, shuffle=True),
    max_steps=150,
    hooks=None
)

eval_spec = tf.estimator.EvalSpec(
    input_fn = lambda: input_fn(valid_X, valid_y, 64),
)

tf.estimator.train_and_evaluate(
    estimator=estimator,
    train_spec=train_spec,
    eval_spec=eval_spec,
)

