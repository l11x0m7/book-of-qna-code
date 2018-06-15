# -*- encoding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ..base import activations
from ..base.activations import Sigmoid, Identity
from ..base.initializers import xavier_uniform_initializer
from ..layers.layer import Layer, Input
from ..base.activations import Softmax as sm


class FullyConnected(Layer):
    def __init__(self, output_dim, input_dim=None, activator='sigmoid',
                    initializer=xavier_uniform_initializer):
        """
        全连接层，即Dense层

        # Params
        output_dim: 输出的维度，即当前层的神经元个数
        input_dim: 输入的维度，即前一层的神经元个数
        activator: 激活函数（如tanh、sigmoid或relu）
        initializer: 初始化方法
        """
        super(FullyConnected, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activator = activations.get(activator)
        self.initializer = initializer
        self.input_shape = None
        self.output_shape = None
        if self.input_dim is not None:
            self.connection(None)

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def delta_W(self):
        return self.__delta_W

    @property
    def delta_b(self):
        return self.__delta_b

    @property
    def delta(self):
        return self.__delta

    @W.setter
    def W(self, W):
        self.__W = W

    @b.setter
    def b(self, b):
        self.__b = b

    @delta_W.setter
    def delta_W(self, delta_W):
        self.__delta_W = delta_W

    @delta_b.setter
    def delta_b(self, delta_b):
        self.__delta_b = delta_b

    @delta.setter
    def delta(self, delta):
        self.__delta = delta

    @property
    def params(self):
        return [self.W, self.b]

    @property
    def grads(self):
        return [self.delta_W, self.delta_b]

    def call(self, pre_layer=None, *args, **kwargs):
        self.connection(pre_layer)
        return self

    def connection(self, pre_layer):
        self.pre_layer = pre_layer
        if pre_layer is None:
            if self.input_dim is None:
                raise ValueError('input_size must not be `None` as the first layer.')
            self.input_shape = [None, self.input_dim]
            self.output_shape = [None, self.output_dim]
        else:
            pre_layer.next_layer = self
            self.input_dim = pre_layer.output_shape[1]
            self.input_shape = pre_layer.output_shape
            self.output_shape = [self.input_shape[0], self.output_dim]
        self.W = self.initializer([self.output_dim, self.input_dim])
        self.b = self.initializer([self.output_dim])
        self.delta_W = np.zeros([self.output_dim, self.input_dim])
        self.delta_b = np.zeros([self.output_dim])
        self.delta = np.zeros([self.input_dim])

    def forward(self, inputs, *args, **kwargs):
        """
        全连接层的前向传播

        # Params
        inputs: 二维矩阵，行代表一个batch的大小，列代表特征

        # Return: 经过该层的输出结果
        """
        inputs = np.asarray(inputs)
        if len(inputs.shape) == 1:
            inputs = inputs[None,:]
        assert list(self.input_shape[1:]) == list(inputs.shape[1:])
        self.input_shape = inputs.shape
        self.output_shape[0] = self.input_shape[0]
        self.inputs = inputs
        # 关键操作，主要是输入与参数矩阵W、偏置b之间的操作，以及激活函数
        self.logit = np.dot(self.inputs, self.W.T) + self.b
        self.output = self.activator.forward(self.logit)
        return self.output

    def backward(self, pre_delta, *args, **kwargs):
        """
        梯度更新
        """
        if len(pre_delta.shape) == 1:
            pre_delta = pre_delta[None,:]
        batch_size, _ = self.inputs.shape
        act_delta = pre_delta * self.activator.backward(self.logit)
        # here should calulate the average value of batch
        self.delta_W = np.dot(act_delta.T, self.inputs)
        self.delta_b = np.mean(act_delta, axis=0)
        self.delta = np.dot(act_delta, self.W)
        return self.delta


Dense = FullyConnected


class Softmax(Dense):
    def __init__(self, output_dim, input_dim=None,
                    initializer=xavier_uniform_initializer):
        super(Softmax, self).__init__(output_dim=output_dim, input_dim=input_dim,
                                      activator='softmax', initializer=initializer)


class Flatten(Layer):
    def __init__(self):
        """
        将输入的多维tensor展开成向量
        """
        super(Flatten, self).__init__()

    @property
    def params(self):
        return list()

    @property
    def grads(self):
        return list()

    def call(self, pre_layer=None, *args, **kwargs):
        self.connection(pre_layer)
        return self

    def connection(self, pre_layer):
        if pre_layer is None:
            raise ValueError('Flatten could not be used as the first layer')
        self.pre_layer = pre_layer
        pre_layer.next_layer = self
        self.input_shape = pre_layer.output_shape
        self.output_shape = self._compute_output_shape(self.input_shape)

    def forward(self, input, *args, **kwargs):
        self.input_shape = input.shape
        self.output_shape = self._compute_output_shape(self.input_shape)
        return np.reshape(input, self.output_shape)

    def backward(self, pre_delta, *args, **kwargs):
        return np.reshape(pre_delta, self.input_shape)

    def _compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "Flatten" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '. '
                             'Make sure to pass a complete "input_shape" '
                             'or "batch_input_shape" argument to the first '
                             'layer in your model.')
        return (input_shape[0], np.prod(input_shape[1:]))


class Dropout(Layer):
    def __init__(self, dropout=0., axis=None):
        """
        Dropout层

        # Params
        dropout: dropout的概率
        axis: 沿某个维度axis进行dropout操作，如果为None则是对所有元素进行
        """
        super(Dropout, self).__init__()
        self.dropout = dropout
        self.axis = axis
        self.mask = None

    @property
    def params(self):
        return list()

    @property
    def grads(self):
        return list()

    def call(self, pre_layer=None, *args, **kwargs):
        self.connection(pre_layer)
        return self

    def connection(self, pre_layer):
        if pre_layer is None:
            raise ValueError('Dropout could not be used as the first layer')
        if self.axis is None:
            self.axis = range(len(pre_layer.output_shape))
        self.output_shape = pre_layer.output_shape
        self.pre_layer = pre_layer
        pre_layer.next_layer = self

    def forward(self, inputs, is_train=True, *args, **kwargs):
        self.input = inputs
        if 0. < self.dropout < 1:
            if is_train:
                self.mask = np.random.binomial(1, 1 - self.dropout, np.asarray(self.input.shape)[self.axis])
                return self.mask * self.input / (1 - self.dropout)
            else:
                return self.input * (1 - self.dropout)
        else:
            return self.input

    def backward(self, pre_delta, *args, **kwargs):
        if 0. < self.dropout < 1.:
            return self.mask * pre_delta
        else:
            return pre_delta


class Activation(Layer):
    def __init__(self, activator, input_shape=None):
        """
        将激活函数应用到输入，得到经过激活函数的输出output

        # Params
        activator: 激活函数名
        """
        super(Activation, self).__init__()
        self.activator = activations.get(activator)
        self.input_shape = input_shape

    @property
    def params(self):
        return list()

    @property
    def grads(self):
        return list()

    def call(self, pre_layer=None, *args, **kwargs):
        self.connection(pre_layer)
        return self

    def connection(self, pre_layer):
        self.pre_layer = pre_layer
        if pre_layer is None:
            if self.input_shape is None:
                raise ValueError('input_size must not be `None` as the first layer.')
            self.output_shape = self.input_shape
        else:
            pre_layer.next_layer = self
            self.input_shape = pre_layer.output_shape
            self.output_shape = self.input_shape

    def forward(self, inputs, *args, **kwargs):
        """
        前向传播
        # Params
        inputs: 2-D tensors, row represents samples, col represents features

        # Return
        None
        """
        inputs = np.asarray(inputs)
        if len(inputs.shape) == 1:
            inputs = inputs[None,:]
        assert list(self.input_shape[1:]) == list(inputs.shape[1:])
        self.input_shape = inputs.shape
        self.output_shape = self.input_shape
        self.inputs = inputs
        self.output = self.activator.forward(self.inputs)
        return self.output

    def backward(self, pre_delta, *args, **kwargs):
        if len(pre_delta.shape) == 1:
            pre_delta = pre_delta[None,:]
        act_delta = pre_delta * self.activator.backward(self.inputs)
        return act_delta
