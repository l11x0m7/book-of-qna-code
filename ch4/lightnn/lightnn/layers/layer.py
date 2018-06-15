# -*- encoding:utf-8 -*-

import numpy as np


class Layer(object):
    def __init__(self):
        self.__input_shape = None
        self.__output_shape = None
        self.__pre_layer = None
        self.__next_layer = None

    @property
    def input_shape(self):
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape):
        self.__input_shape = input_shape

    @property
    def output_shape(self):
        return self.__output_shape

    @output_shape.setter
    def output_shape(self, output_shape):
        self.__output_shape = output_shape

    @property
    def pre_layer(self):
        return self.__pre_layer

    @pre_layer.setter
    def pre_layer(self, pre_layer):
        self.__pre_layer = pre_layer

    @property
    def next_layer(self):
        return self.__next_layer

    @next_layer.setter
    def next_layer(self, next_layer):
        self.__next_layer = next_layer

    def connection(self, pre_layer):
        raise NotImplementedError('function `connection` should be implemented')

    @property
    def params(self):
        raise NotImplementedError('function `get_params` should be implemented')

    @property
    def grads(self):
        raise NotImplementedError('function `get_grads` should be implemented')

    def forward(self, inputs, *args, **kwargs):
        raise NotImplementedError('function `forward` should be implemented')

    def backward(self, pre_delta, *args, **kwargs):
        raise NotImplementedError('function `backward` should be implemented')

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError('function `call` should be implemented')


class Input(Layer):
    def __init__(self, input_shape=None, batch_size=None,
                 batch_input_shape=None, input_tensor=None,
                 dtype=None):
        super(Input, self).__init__()
        if input_shape and batch_input_shape:
            raise ValueError('Only provide the input_shape OR '
                             'batch_input_shape argument to '
                             'Input, not both at the same time.')
        if input_tensor is not None and batch_input_shape is None:
            # If input_tensor is set, and batch_input_shape is not set:
            # Attempt automatic input shape inference.
            try:
                batch_input_shape = np.asarray(input_tensor).shape
            except TypeError:
                if not input_shape and not batch_input_shape:
                    raise ValueError('Input was provided '
                                     'an input_tensor argument, '
                                     'but its input shape cannot be '
                                     'automatically inferred. '
                                     'You should pass an input_shape or '
                                     'batch_input_shape argument.')
        if not batch_input_shape:
            if not input_shape:
                raise ValueError('An Input layer should be passed either '
                                 'a `batch_input_shape` or an `input_shape`.')
            else:
                if isinstance(input_shape, np.int):
                    batch_input_shape = (batch_size, input_shape)
                else:
                    batch_input_shape = (batch_size,) + tuple(input_shape)
        else:
            batch_input_shape = tuple(batch_input_shape)

        if not dtype:
            if input_tensor is None:
                dtype = np.float32
            else:
                dtype = np.dtype(input_tensor)

        self.input_shape = batch_input_shape
        self.dtype = dtype
        self.connection(None)

    @property
    def params(self):
        return list()

    @property
    def grads(self):
        return list()

    def connection(self, pre_layer):
        if pre_layer is not None:
            raise ValueError('Input layer must be your first layer.')
        self.output_shape = self.input_shape

    def call(self, pre_layer=None, *args, **kwargs):
        raise ValueError('Input layer is not callable.')

    def forward(self, inputs, *args, **kwargs):
        inputs = np.asarray(inputs)
        assert self.input_shape[1:] == inputs.shape[1:]
        self.input_shape = inputs.shape
        self.output_shape = self.input_shape
        return inputs

    def backward(self, pre_delta, *args, **kwargs):
        pass
