#-*- encoding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ..base.activations import Sigmoid, Softmax
from ..base.initializers import xavier_uniform_initializer
from ..layers.core import FullyConnected


### !!!!!!!!!!!!! Deprecated file !!!!!!!!!!!!! ###
class NetWork(object):
    def __init__(self, sizes, cost='bce', activator='sigmoid', initializer=xavier_uniform_initializer, lr=1e-1, lmbda=None):
        self.sizes = sizes
        self.layer_num = len(sizes)
        self.cost = cost
        self.lr = lr
        self.lmbda = lmbda
        self.layers = [FullyConnected(i_size, o_size,
                            activator, initializer) for i_size, o_size
                                in zip(sizes[:-1], sizes[1:])]
        if self.layers[-1].output_dim > 2:
            self.layers[-1].activator = Softmax

    def train(self, input_data, input_label, epoch, batch_size, verbose=True):
        for ep in xrange(epoch):
            batch_epoch = len(input_data) // batch_size + 1
            for mini_batch, batch_label in self.__batch_sample(batch_epoch, batch_size, input_data, input_label):
                self.train_batch(mini_batch, batch_label, len(input_data))

            if verbose:
                print("Epoch %s training complete" % ep)
                cost = self.total_cost(input_data, input_label)
                print("Cost on training data: {}".format(cost))
                accuracy = self.accuracy(input_data, input_label)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, len(input_data)))

    def train_batch(self, mini_batch, batch_label, training_set_n):
        batch_delta_W = [np.zeros(layer.W.shape) for layer in self.layers]
        batch_delta_b = [np.zeros(layer.b.shape) for layer in self.layers]
        for x, y in zip(mini_batch, batch_label):
            self.feedforward(x)
            self.backprop(y)
            for layer_num in xrange(self.layer_num-1):
                batch_delta_W[layer_num] += self.layers[layer_num].delta_W
                batch_delta_b[layer_num] += self.layers[layer_num].delta_b


        # print batch_delta_W
        for i, layer in enumerate(self.layers):
            update_delta_W = batch_delta_W[i] * self.lr / len(mini_batch)
            update_delta_b = batch_delta_b[i] * self.lr / len(mini_batch)
            if self.lmbda is not None:
                update_delta_W += self.lmbda * self.lr * layer.W / training_set_n
            layer.step(update_delta_W, update_delta_b)


    def backprop(self, y):
        """
        Calculate the gradient of each parameter
        :param x: single input data
        :param y: single one hot output label
        :return: None
        """
        delta = self.cost.backward(self.layers[-1].output, y)
        for layer in self.layers[::-1]:
            delta = layer.backward(pre_delta=delta)

    def feedforward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
            print(output)
            print(layer.W)
        return output

    def __batch_sample(self, batch_epoch, batch_size, input_data, input_label):
        for _ in xrange(batch_epoch):
            choice = np.random.choice(len(input_data), batch_size, replace=False)
            yield np.asarray(input_data)[choice], np.asarray(input_label)[choice]


    def accuracy(self, data, label):
        results = 0
        for x, y in zip(data, label):
            self.feedforward(x)
            a = self.layers[-1].output
            results += (np.argmax(a) == np.argmax(y))
        return results


    def total_cost(self, data, label):
        cost = 0.0
        for x, y in zip(data, label):
            self.feedforward(x)
            a = self.layers[-1].output
            cost += self.cost.forward(a, y) / len(data)
        if self.lmbda is not None:
            cost += 0.5 * (self.lmbda / len(data)) * sum(
                np.linalg.norm(layer.W)**2 for layer in self.layers)
        return cost
