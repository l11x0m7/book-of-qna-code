# -*- encoding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from lightnn.models.models import Sequential, Model
from lightnn.layers.core import Dense, Flatten, Softmax, Input, Dropout, Activation
from lightnn.layers.convolutional import Conv2d
from lightnn.layers.pooling import MaxPooling, AvgPooling
from lightnn.base.activations import Relu, Selu
from lightnn.base.optimizers import SGD, Momentum, RMSProp, Adam, Adagrad, Adadelta


def mlp_random():
    """test MLP with random data and Sequential

    """
    input_size = 600
    input_dim = 20
    label_size = 10
    train_X = np.random.random((input_size, input_dim))
    train_y = np.zeros((input_size, label_size))
    for _ in xrange(input_size):
        train_y[_,np.random.randint(0, label_size)] = 1

    model = Sequential()
    model.add(Input(input_shape=input_dim))
    model.add(Dense(100, activator='selu'))
    model.add(Softmax(label_size))
    model.compile('CCE')
    model.fit(train_X, train_y, verbose=1)


def mlp_mnist():
    """test MLP with MNIST data and Sequential

    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
    training_data = np.array([image.flatten() for image in mnist.train.images])
    training_label = mnist.train.labels
    valid_data = np.array([image.flatten() for image in mnist.validation.images])
    valid_label = mnist.validation.labels
    input_dim = training_data.shape[1]
    label_size = training_label.shape[1]

    model = Sequential()
    model.add(Input(input_shape=(input_dim, )))
    model.add(Dense(300, activator='selu'))
    model.add(Dropout(0.2))
    model.add(Softmax(label_size))
    model.compile('CCE', optimizer=SGD())
    model.fit(training_data, training_label, validation_data=(valid_data, valid_label))


def cnn_random():
    """test CNN with random data and Sequential

    """
    input_size = 600
    input_dim = 28
    input_depth = 1
    label_size = 10
    train_X = np.random.random((input_size, input_dim, input_dim, input_depth))
    train_y = np.zeros((input_size, label_size))
    for _ in xrange(input_size):
        train_y[_,np.random.randint(0, label_size)] = 1

    model =Sequential()
    model.add(Input(batch_input_shape=(None, 28, 28, 1)))
    model.add(Conv2d((3, 3), 1, activator='relu'))
    model.add(AvgPooling((2, 2), stride=2))
    model.add(Conv2d((4, 4), 2, activator='relu'))
    model.add(AvgPooling((2, 2), stride=2))
    model.add(Flatten())
    model.add(Softmax(label_size))
    model.compile('CCE', optimizer=SGD(1e-2))
    model.fit(train_X, train_y)


def cnn_mnist():
    """test CNN with MNIST data and Sequential

    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
    training_data = np.array([image.reshape(28, 28, 1) for image in mnist.train.images])
    training_label = mnist.train.labels
    valid_data = np.array([image.reshape(28, 28, 1) for image in mnist.validation.images])
    valid_label = mnist.validation.labels
    label_size = training_label.shape[1]

    model =Sequential()
    model.add(Input(batch_input_shape=(None, 28, 28, 1)))
    model.add(Conv2d((3, 3), 1, activator='selu'))
    model.add(AvgPooling((2, 2), stride=2))
    model.add(Conv2d((4, 4), 2, activator='selu'))
    model.add(AvgPooling((2, 2), stride=2))
    model.add(Flatten())
    model.add(Softmax(label_size))
    model.compile('CCE', optimizer=SGD(lr=1e-2))
    model.fit(training_data, training_label, validation_data=(valid_data, valid_label), verbose=2)


def model_mlp_random():
    """test MLP with random data and Model

    """
    input_size = 600
    input_dim = 20
    label_size = 10
    train_X = np.random.random((input_size, input_dim))
    train_y = np.zeros((input_size, label_size))
    for _ in xrange(input_size):
        train_y[_,np.random.randint(0, label_size)] = 1

    input = Input(input_shape=input_dim)
    d1 = Dense(100, activator='selu')(input)
    s1 = Softmax(label_size)(d1)
    model = Model(input, s1)
    model.compile('CCE')
    model.fit(train_X, train_y, verbose=1)


def model_mlp_mnist():
    """test MLP with MNIST data and Model

    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
    training_data = np.array([image.flatten() for image in mnist.train.images])
    training_label = mnist.train.labels
    valid_data = np.array([image.flatten() for image in mnist.validation.images])
    valid_label = mnist.validation.labels
    input_dim = training_data.shape[1]
    label_size = training_label.shape[1]

    dense_1 = Dense(300, input_dim=input_dim, activator=None)
    dense_2 = Activation('selu')(dense_1)
    dropout_1 = Dropout(0.2)(dense_2)
    softmax_1 = Softmax(label_size)(dropout_1)
    model = Model(dense_1, softmax_1)
    model.compile('CCE', optimizer=Adadelta())
    model.fit(training_data, training_label, validation_data=(valid_data, valid_label))


if __name__ == '__main__':
    # mlp_random()
    # mlp_mnist()
    # cnn_random()
    # cnn_mnist()
    # model_mlp_random()
    model_mlp_mnist()
