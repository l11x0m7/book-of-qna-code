# -*- encoding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .layer import Layer
from ..base import initializers
from ..base import activations


class Recurrent(Layer):
    """
    Abstract base class for recurrent layers.

    Do not use in a model -- it's not a valid layer!

    Use its children classes `LSTM`, `GRU` and `SimpleRNN` instead.
    All recurrent layers (`LSTM`, `GRU`, `SimpleRNN`) also
    follow the specifications of this class and accept
    the keyword arguments listed below.
    """
    def __init__(self, output_dim,
                 input_shape=None,
                 return_sequences=False,
                 **kwargs):
        # input_shape:(batch size, sequence length, input dimension)
        super(Recurrent, self).__init__()
        self.output_dim = output_dim
        self.return_sequences = return_sequences
        self.input_shape = input_shape
        self.output_shape = None
        if input_shape is not None:
            self.input_dim = input_shape[-1]

    def connection(self, pre_layer):
        if pre_layer is None:
            if self.input_shape is None:
                raise ValueError('input_shape must not be `None` as the first layer.')
        else:
            self.input_shape = pre_layer.output_shape
            self.pre_layer = pre_layer
            pre_layer.next_layer = self

        self.input_dim = self.input_shape[-1]

        if self.return_sequences:
            self.output_shape = [self.input_shape[0], self.input_shape[1], self.output_dim]
        else:
            self.output_shape = [self.input_shape[0], self.output_dim]


class SimpleRNN(Recurrent):
    """
    Simple RNN unit.

    Fully-connected RNN where the output is to be fed back to input.
    """

    def __init__(self, output_dim,
                 input_shape=None,
                 activator='tanh',
                 kernel_initializer='glorot_uniform_initializer',
                 recurrent_initializer='orthogonal_initializer',
                 bias_initializer='zeros',
                 use_bias=True,
                 return_sequences=False,
                 **kwargs):
        """
        基本的循环神经网络层

        # Params
        output_dim: 输出的维度，即当前层的神经元个数
        input_shape: 输入的形状
        activator: 激活函数
        kernel_initializer: 输入与隐层的参数初始化方法
        recurrent_initializer: 循环内部的参数初始化方法
        bias_initializer: 偏置bias初始化方法
        use_bias: 是否加偏置bias
        return_sequences: 表示是否返回一整个序列
        """
        super(SimpleRNN, self).__init__(output_dim, input_shape, return_sequences, **kwargs)
        self.activator = activations.get(activator)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.use_bias = use_bias
        self.W = None
        self.U = None
        self.b = None
        self.delta_W = None
        self.delta_U = None
        self.delta_b = None
        if input_shape is not None:
            self.connection(None)

    @property
    def W(self):
        return self.__W

    @W.setter
    def W(self, W):
        self.__W = W

    @property
    def U(self):
        return self.__U

    @U.setter
    def U(self, U):
        self.__U = U

    @property
    def b(self):
        return self.__b

    @b.setter
    def b(self, b):
        self.__b = b

    @property
    def delta_W(self):
        return self.__delta_W

    @delta_W.setter
    def delta_W(self, delta_W):
        self.__delta_W = delta_W

    @property
    def delta_U(self):
        return self.__delta_U

    @delta_U.setter
    def delta_U(self, delta_U):
        self.__delta_U = delta_U

    @property
    def delta_b(self):
        return self.__delta_b

    @delta_b.setter
    def delta_b(self, delta_b):
        self.__delta_b = delta_b

    @property
    def params(self):
        if self.use_bias:
            return [self.W, self.U, self.b]
        return [self.W, self.U]

    @property
    def grads(self):
        if self.use_bias:
            return [self.delta_W, self.delta_U, self.delta_b]
        return [self.delta_W, self.delta_U]

    def call(self, pre_layer=None, *args, **kwargs):
        self.connection(pre_layer)
        return self

    def connection(self, pre_layer):
        super(SimpleRNN, self).connection(pre_layer)
        self.W = self.kernel_initializer((self.input_dim, self.output_dim))
        self.U = self.recurrent_initializer((self.output_dim, self.output_dim))
        if self.use_bias:
            self.b = self.bias_initializer((self.output_dim, ))

    def forward(self, inputs, *args, **kwargs):
        """
        RNN单元的前向传播

        # Params
        inputs: 输入的数据

        # Return: 返回该层的输出
        """
        # clear states
        # inputs: batch_size, time_step, out_dim
        inputs = np.asarray(inputs)
        self.inputs = inputs
        assert list(inputs.shape[1:]) == list(self.input_shape[1:])
        self.input_shape = inputs.shape
        self.output_shape[0] = self.input_shape[0]
        nb_batch, nb_seq, nb_input_dim = self.input_shape
        self.outputs = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.logits = np.zeros((nb_batch, nb_seq, self.output_dim))
        pre_state = np.zeros((nb_batch, self.output_dim))
        # 核心部分，对于每个时间步骤time step，根据当前输入与前一时刻输出，产生当前时刻输出
        for t in xrange(nb_seq):
            self.outputs[:,t,:] = self.inputs[:,t,:].dot(self.W) + pre_state.dot(self.U)
            if self.use_bias:
                self.outputs[:,t,:] += self.b
            self.logits[:,t,:] = self.outputs[:,t,:]
            self.outputs[:,t,:] = self.activator.forward(self.outputs[:,t,:])
            pre_state = self.outputs[:,t,:]

        if self.return_sequences:
            return self.outputs
        else:
            return self.outputs[:,-1,:]

    def backward(self, pre_delta, *args, **kwargs):
        self.delta_W = np.zeros(self.W.shape)
        self.delta_U = np.zeros(self.U.shape)
        if self.use_bias:
            self.delta_b = np.zeros(self.b.shape)
        nb_batch, nb_seq, nb_input_dim = self.input_shape
        nb_output_dim = self.output_dim
        self.delta = np.zeros(self.input_shape)
        if self.return_sequences:
            assert len(pre_delta.shape) == 3
            # 同一层的误差传递（从T到1）,此处的time_delta为delta_E/delta_z
            time_delta = pre_delta[:,nb_seq-1,:] * self.activator.backward(self.logits[:,nb_seq-1,:])
        else:
            assert len(pre_delta.shape) == 2
            # 同一层的误差传递（从T到1）,此处的time_delta为delta_E/delta_z
            time_delta = pre_delta * self.activator.backward(self.logits[:,-1,:])
        for t in xrange(nb_seq - 1, -1, -1):
            pre_logit = np.zeros((nb_batch, nb_output_dim)) if t == 0 else self.logits[:,t - 1,:]
            cur_input = self.inputs[:,t,:]
            pre_state = np.zeros((nb_batch, nb_output_dim)) if t == 0 else self.outputs[:,t - 1,:]
            # 求U的梯度
            self.delta_U += np.dot(pre_state.T, time_delta) / nb_batch
            # 求W的梯度
            self.delta_W += np.dot(cur_input.T, time_delta) / nb_batch
            # 求b的梯度
            if self.use_bias:
                self.delta_b += np.mean(time_delta, axis=0)
            # 求传到上一层的误差,layerwise
            self.delta[:,t,:] = np.dot(time_delta, self.W.T)
            # 求同一层不同时间的误差,timewise
            if t > 0:
                # 下面两种计算同层不同时间误差的方法等效
                # 方法1
                # time_delta = np.asarray(
                #     map(
                #     np.dot, *(time_delta, np.asarray(map(
                #     lambda logit:(self.activator.backward(logit) * self.U.T)
                #     , pre_logit)))
                #     )
                # )
                # 方法2
                # for bn in xrange(nb_batch):
                #     time_delta[bn,:] = np.dot(
                #     time_delta[bn,:], np.dot(
                #     np.diag(self.activator.backward(pre_logit[bn])),
                #             self.U).T)
                # 方法3,效率相对比较高
                time_delta = np.sum(time_delta[:,None,:].transpose((0,2,1)) * \
                                (self.activator.backward(pre_logit)[:,None,:] * self.U.T), axis=1)
                if self.return_sequences:
                    time_delta += pre_delta[:,t - 1,:] * \
                             self.activator.backward(pre_logit)
        return self.delta


class LSTM(Recurrent):
    """
    Long-Short Term Memory unit - Hochreiter 1997.

   For a step-by-step description of the algorithm, see
   [this tutorial](http://deeplearning.net/tutorial/lstm.html).

   References:
   1. LSTM: A Search Space Odyssey
      (https://arxiv.org/pdf/1503.04069.pdf)
   2. Backpropogating an LSTM: A Numerical Example
      (https://blog.aidangomez.ca/2016/04/17/
                Backpropogating-an-LSTM-A-Numerical-Example/)
   3. LSTM(https://github.com/nicodjimenez/lstm)
    """

    def __init__(self, output_dim,
                 input_shape=None,
                 activator='tanh',
                 recurrent_activator='sigmoid',
                 kernel_initializer='glorot_uniform_initializer',
                 recurrent_initializer='orthogonal_initializer',
                 bias_initializer='zeros',
                 use_bias=True,
                 return_sequences=False,
                 **kwargs):
        """
        LSTM单元

        # Params
        output_dim: 输出的维度，即当前层的神经元个数
        input_shape: 输入的形状
        activator: 激活函数
        kernel_initializer: 输入与隐层的参数初始化方法
        recurrent_initializer: 循环内部的参数初始化方法
        bias_initializer: 偏置bias初始化方法
        use_bias: 是否加偏置bias
        return_sequences: 表示是否返回一整个序列
        """
        super(LSTM, self).__init__(output_dim, input_shape, return_sequences, **kwargs)
        self.use_bias = use_bias
        self.activator = activations.get(activator)
        self.recurrent_activator = activations.get(recurrent_activator)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # input gate
        self.__W_i = None
        self.__U_i = None
        self.__b_i = None
        self.__delta_W_i = None
        self.__delta_U_i = None
        self.__delta_b_i = None

        # forget gate
        self.__W_f = None
        self.__U_f = None
        self.__b_f = None
        self.__delta_W_f = None
        self.__delta_U_f = None
        self.__delta_b_f = None

        # output gate
        self.__W_o = None
        self.__U_o = None
        self.__b_o = None
        self.__delta_W_o = None
        self.__delta_U_o = None
        self.__delta_b_o = None

        # cell memory(long term)
        self.__W_c = None
        self.__U_c = None
        self.__b_c = None
        self.__delta_W_c = None
        self.__delta_U_c = None
        self.__delta_b_c = None

        if input_shape is not None:
            self.connection(None)

    @property
    def W_i(self):
        return self.__W_i

    @W_i.setter
    def W_i(self, W_i):
        self.__W_i = W_i

    @property
    def U_i(self):
        return self.__U_i

    @U_i.setter
    def U_i(self, U_i):
        self.__U_i = U_i

    @property
    def b_i(self):
        return self.__b_i

    @b_i.setter
    def b_i(self, b_i):
        self.__b_i = b_i

    @property
    def delta_W_i(self):
        return self.__delta_W_i

    @delta_W_i.setter
    def delta_W_i(self, delta_W_i):
        self.__delta_W_i = delta_W_i

    @property
    def delta_U_i(self):
        return self.__delta_U_i

    @delta_U_i.setter
    def delta_U_i(self, delta_U_i):
        self.__delta_U_i = delta_U_i

    @property
    def delta_b_i(self):
        return self.__delta_b_i

    @delta_b_i.setter
    def delta_b_i(self, delta_b_i):
        self.__delta_b_i = delta_b_i

    @property
    def W_o(self):
        return self.__W_o

    @W_o.setter
    def W_o(self, W_o):
        self.__W_o = W_o

    @property
    def U_o(self):
        return self.__U_o

    @U_o.setter
    def U_o(self, U_o):
        self.__U_o = U_o

    @property
    def b_o(self):
        return self.__b_o

    @b_o.setter
    def b_o(self, b_o):
        self.__b_o = b_o

    @property
    def delta_W_o(self):
        return self.__delta_W_o

    @delta_W_o.setter
    def delta_W_o(self, delta_W_o):
        self.__delta_W_o = delta_W_o

    @property
    def delta_U_o(self):
        return self.__delta_U_o

    @delta_U_o.setter
    def delta_U_o(self, delta_U_o):
        self.__delta_U_o = delta_U_o

    @property
    def delta_b_o(self):
        return self.__delta_b_o

    @delta_b_o.setter
    def delta_b_o(self, delta_b_o):
        self.__delta_b_o = delta_b_o

    @property
    def W_f(self):
        return self.__W_f

    @W_f.setter
    def W_f(self, W_f):
        self.__W_f = W_f

    @property
    def U_f(self):
        return self.__U_f

    @U_f.setter
    def U_f(self, U_f):
        self.__U_f = U_f

    @property
    def b_f(self):
        return self.__b_f

    @b_f.setter
    def b_f(self, b_f):
        self.__b_f = b_f

    @property
    def delta_W_f(self):
        return self.__delta_W_f

    @delta_W_f.setter
    def delta_W_f(self, delta_W_f):
        self.__delta_W_f = delta_W_f

    @property
    def delta_U_f(self):
        return self.__delta_U_f

    @delta_U_f.setter
    def delta_U_f(self, delta_U_f):
        self.__delta_U_f = delta_U_f

    @property
    def delta_b_f(self):
        return self.__delta_b_f

    @delta_b_f.setter
    def delta_b_f(self, delta_b_f):
        self.__delta_b_f = delta_b_f

    @property
    def W_c(self):
        return self.__W_c

    @W_c.setter
    def W_c(self, W_c):
        self.__W_c = W_c

    @property
    def U_c(self):
        return self.__U_c

    @U_c.setter
    def U_c(self, U_c):
        self.__U_c = U_c

    @property
    def b_c(self):
        return self.__b_c

    @b_c.setter
    def b_c(self, b_c):
        self.__b_c = b_c

    @property
    def delta_W_c(self):
        return self.__delta_W_c

    @delta_W_c.setter
    def delta_W_c(self, delta_W_c):
        self.__delta_W_c = delta_W_c

    @property
    def delta_U_c(self):
        return self.__delta_U_c

    @delta_U_c.setter
    def delta_U_c(self, delta_U_c):
        self.__delta_U_c = delta_U_c

    @property
    def delta_b_c(self):
        return self.__delta_b_c

    @delta_b_c.setter
    def delta_b_c(self, delta_b_c):
        self.__delta_b_c = delta_b_c

    @property
    def params(self):
        if self.use_bias:
            return [self.W_i, self.U_i, self.b_i,
                    self.W_o, self.U_o, self.b_o,
                    self.W_f, self.U_f, self.b_f,
                    self.W_c, self.U_c, self.b_c]
        return [self.W_i, self.U_i,
                self.W_o, self.U_o,
                self.W_f, self.U_f,
                self.W_c, self.U_c]

    @property
    def grads(self):
        if self.use_bias:
            return [self.delta_W_i, self.delta_U_i, self.delta_b_i,
                    self.delta_W_o, self.delta_U_o, self.delta_b_o,
                    self.delta_W_f, self.delta_U_f, self.delta_b_f,
                    self.delta_W_c, self.delta_U_c, self.delta_b_c]
        return [self.delta_W_i, self.delta_U_i,
                self.delta_W_o, self.delta_U_o,
                self.delta_W_f, self.delta_U_f,
                self.delta_W_c, self.delta_U_c]

    def call(self, pre_layer=None, *args, **kwargs):
        self.connection(pre_layer)
        return self

    def connection(self, pre_layer):
        super(LSTM, self).connection(pre_layer)
        self.W_i = self.kernel_initializer((self.input_dim, self.output_dim))
        self.U_i = self.recurrent_initializer((self.output_dim, self.output_dim))

        self.W_o = self.kernel_initializer((self.input_dim, self.output_dim))
        self.U_o = self.recurrent_initializer((self.output_dim, self.output_dim))

        self.W_f = self.kernel_initializer((self.input_dim, self.output_dim))
        self.U_f = self.recurrent_initializer((self.output_dim, self.output_dim))

        self.W_c = self.kernel_initializer((self.input_dim, self.output_dim))
        self.U_c = self.recurrent_initializer((self.output_dim, self.output_dim))

        if self.use_bias:
            self.b_i = self.bias_initializer((self.output_dim, ))
            self.b_o = self.bias_initializer((self.output_dim, ))
            self.b_f = self.bias_initializer((self.output_dim, ))
            self.b_c = self.bias_initializer((self.output_dim, ))

    def forward(self, inputs, *args, **kwargs):
        # inputs: batch_size, time_step, out_dim
        inputs = np.asarray(inputs)
        self.inputs = inputs
        assert list(inputs.shape[1:]) == list(self.input_shape[1:])
        self.input_shape = inputs.shape
        self.output_shape[0] = self.input_shape[0]
        nb_batch, nb_seq, nb_input_dim = self.input_shape

        self.cells = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.outputs = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.logits_i = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.logits_o = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.logits_f = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.logits_c = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.outputs_i = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.outputs_o = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.outputs_f = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.outputs_c = np.zeros((nb_batch, nb_seq, self.output_dim))
        for t in xrange(nb_seq):
            h_pre = np.zeros((nb_batch, self.output_dim)) \
                        if t == 0 else self.outputs[:, t - 1, :]
            c_pre = np.zeros((nb_batch, self.output_dim)) \
                        if t == 0 else self.cells[:, t - 1, :]
            x_now = self.inputs[:,t,:]
            i = x_now.dot(self.W_i) + h_pre.dot(self.U_i)
            o = x_now.dot(self.W_o) + h_pre.dot(self.U_o)
            f = x_now.dot(self.W_f) + h_pre.dot(self.U_f)
            c_tilde = x_now.dot(self.W_c) + h_pre.dot(self.U_c)

            if self.use_bias:
                i += self.b_i
                o += self.b_o
                f += self.b_f
                c_tilde += self.b_c

            self.logits_i[:,t,:] = i
            self.logits_o[:,t,:] = o
            self.logits_f[:,t,:] = f
            self.logits_c[:,t,:] = c_tilde

            i = self.recurrent_activator.forward(i)
            o = self.recurrent_activator.forward(o)
            f = self.recurrent_activator.forward(f)
            c_tilde = self.activator.forward(c_tilde)

            self.outputs_i[:,t,:] = i
            self.outputs_o[:,t,:] = o
            self.outputs_f[:,t,:] = f
            self.outputs_c[:,t,:] = c_tilde

            self.cells[:,t,:] = f * c_pre + i * c_tilde
            self.outputs[:,t,:] = o * self.activator.forward(self.cells[:,t,:])

        if self.return_sequences:
            return self.outputs
        else:
            return self.outputs[:,-1,:]

    def backward(self, pre_delta, *args, **kwargs):
        """
        BPTT.

        思路步骤:
        1. 求出loss关于h_{t-1},i_t,o_t,f_t和c_tilde_t的logit（激活函数里面的线性和）的偏导
        2. 所有其余的偏导（W,U,b,以及往前一个timestep和往前一层传的偏导）都可以由上面的几个偏导表示
        3. 注意loss关于c_t的偏导,包含两个:一个是c_{t+1}中的c_t,一个是h_t中的c_t

        # Params
        pre_delta: gradients from the next layer.

        # Return
        gradients to the previous layer.
        """
        self.delta_W_i = np.zeros(self.W_i.shape)
        self.delta_W_o = np.zeros(self.W_o.shape)
        self.delta_W_f = np.zeros(self.W_f.shape)
        self.delta_W_c = np.zeros(self.W_c.shape)
        self.delta_U_i = np.zeros(self.U_i.shape)
        self.delta_U_o = np.zeros(self.U_o.shape)
        self.delta_U_f = np.zeros(self.U_f.shape)
        self.delta_U_c = np.zeros(self.U_c.shape)
        if self.use_bias:
            self.delta_b_i = np.zeros(self.b_i.shape)
            self.delta_b_o = np.zeros(self.b_o.shape)
            self.delta_b_f = np.zeros(self.b_f.shape)
            self.delta_b_c = np.zeros(self.b_c.shape)

        nb_batch, nb_seq, nb_input_dim = self.input_shape
        nb_output_dim = self.output_dim
        self.delta = np.zeros(self.input_shape)
        if self.return_sequences:
            assert len(pre_delta.shape) == 3
            # 此处的time_delta为delta_E/delta_output
            time_delta = pre_delta[:,nb_seq - 1,:]
        else:
            assert len(pre_delta.shape) == 2
            # 此处的time_delta为delta_E/delta_output
            time_delta = pre_delta
        future_state = np.zeros((nb_batch, nb_output_dim))
        for t in np.arange(nb_seq)[::-1]:
            logit_i = self.logits_i[:,t,:]
            logit_o = self.logits_o[:,t,:]
            logit_f = self.logits_f[:,t,:]
            logit_c_tilde = self.logits_c[:,t,:]
            output_i = self.outputs_i[:,t,:]
            output_o = self.outputs_o[:,t,:]
            output_c_tilde = self.outputs_c[:,t,:]
            c = self.cells[:,t,:]
            pre_c = np.zeros((nb_batch, nb_output_dim)) \
                        if t == 0 else self.cells[:,t - 1,:]
            pre_h = np.zeros((nb_batch, nb_output_dim)) \
                        if t == 0 else self.outputs[:,t - 1,:]
            cur_x = self.inputs[:,t,:]
            future_output_f = np.zeros((nb_batch, nb_output_dim)) \
                        if t == nb_seq - 1 else self.outputs_f[:,t + 1,:]
            # cell state
            pre_delta_state = time_delta * output_o * self.activator.backward(c) + \
                                future_state * future_output_f
            pre_delta_i = pre_delta_state * output_c_tilde * self.recurrent_activator.backward(logit_i)
            pre_delta_o = time_delta * self.activator.forward(c) * \
                          self.recurrent_activator.backward(logit_o)
            pre_delta_f = pre_delta_state * pre_c * self.recurrent_activator.backward(logit_f)
            pre_delta_c = pre_delta_state * output_i * self.activator.backward(logit_c_tilde)

            future_state = pre_delta_state

            # 求U的梯度
            self.delta_U_i += np.dot(pre_h.T, pre_delta_i) / nb_batch
            self.delta_U_o += np.dot(pre_h.T, pre_delta_o) / nb_batch
            self.delta_U_f += np.dot(pre_h.T, pre_delta_f) / nb_batch
            self.delta_U_c += np.dot(pre_h.T, pre_delta_c) / nb_batch
            # 求W的梯度
            self.delta_W_i += np.dot(cur_x.T, pre_delta_i) / nb_batch
            self.delta_W_o += np.dot(cur_x.T, pre_delta_o) / nb_batch
            self.delta_W_f += np.dot(cur_x.T, pre_delta_f) / nb_batch
            self.delta_W_c += np.dot(cur_x.T, pre_delta_c) / nb_batch
            # 求b的梯度
            if self.use_bias:
                self.delta_b_i += np.mean(pre_delta_i, axis=0)
                self.delta_b_o += np.mean(pre_delta_o, axis=0)
                self.delta_b_f += np.mean(pre_delta_f, axis=0)
                self.delta_b_c += np.mean(pre_delta_c, axis=0)
            # 求传到上一层的误差,layerwise
            self.delta[:,t,:] = np.dot(pre_delta_i, self.W_i.T) + \
                                np.dot(pre_delta_o, self.W_o.T) + \
                                np.dot(pre_delta_f, self.W_f.T) + \
                                np.dot(pre_delta_c, self.W_c.T)
            # 求同一层不同时间的误差,timewise
            if t > 0:
                time_delta = np.dot(pre_delta_i, self.U_i.T) + \
                             np.dot(pre_delta_o, self.U_o.T) + \
                             np.dot(pre_delta_f, self.U_f.T) + \
                             np.dot(pre_delta_c, self.U_c.T)
                if self.return_sequences:
                    time_delta += pre_delta[:,t - 1,:]
        return self.delta


class GRU(Recurrent):
    """
    Gated Recurrent Unit.

    A variant of LSTM(simplified version)

    References:
    1. LSTM: A Search Space Odyssey
      (https://arxiv.org/pdf/1503.04069.pdf)
    """

    def __init__(self, output_dim,
                 input_shape=None,
                 activator='tanh',
                 recurrent_activator='sigmoid',
                 kernel_initializer='glorot_uniform_initializer',
                 recurrent_initializer='orthogonal_initializer',
                 bias_initializer='zeros',
                 use_bias=True,
                 return_sequences=False,
                 **kwargs):
        """
        GRU单元

        # Params
        output_dim: 输出的维度，即当前层的神经元个数
        input_shape: 输入的形状
        activator: 激活函数
        kernel_initializer: 输入与隐层的参数初始化方法
        recurrent_initializer: 循环内部的参数初始化方法
        bias_initializer: 偏置bias初始化方法
        use_bias: 是否加偏置bias
        return_sequences: 表示是否返回一整个序列
        """
        super(GRU, self).__init__(output_dim, input_shape, return_sequences, **kwargs)
        self.use_bias = use_bias
        self.activator = activations.get(activator)
        self.recurrent_activator = activations.get(recurrent_activator)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # reset gate
        self.__W_r = None
        self.__U_r = None
        self.__b_r = None
        self.__delta_W_r = None
        self.__delta_U_r = None
        self.__delta_b_r = None

        # update gate
        self.__W_z = None
        self.__U_f = None
        self.__b_f = None
        self.__delta_W_f = None
        self.__delta_U_f = None
        self.__delta_b_f = None

        # hidden state
        self.__W_h = None
        self.__U_h = None
        self.__b_h = None
        self.__delta_W_h = None
        self.__delta_U_h = None
        self.__delta_b_h = None

        if input_shape is not None:
            self.connection(None)

    @property
    def W_r(self):
        return self.__W_r

    @W_r.setter
    def W_r(self, W_r):
        self.__W_r = W_r

    @property
    def U_r(self):
        return self.__U_r

    @U_r.setter
    def U_r(self, U_r):
        self.__U_r = U_r

    @property
    def b_r(self):
        return self.__b_r

    @b_r.setter
    def b_r(self, b_r):
        self.__b_r = b_r

    @property
    def delta_W_r(self):
        return self.__delta_W_r

    @delta_W_r.setter
    def delta_W_r(self, delta_W_r):
        self.__delta_W_r = delta_W_r

    @property
    def delta_U_r(self):
        return self.__delta_U_r

    @delta_U_r.setter
    def delta_U_r(self, delta_U_r):
        self.__delta_U_r = delta_U_r

    @property
    def delta_b_r(self):
        return self.__delta_b_r

    @delta_b_r.setter
    def delta_b_r(self, delta_b_r):
        self.__delta_b_r = delta_b_r

    @property
    def W_z(self):
        return self.__W_z

    @W_z.setter
    def W_z(self, W_z):
        self.__W_z = W_z

    @property
    def U_z(self):
        return self.__U_z

    @U_z.setter
    def U_z(self, U_z):
        self.__U_z = U_z

    @property
    def b_z(self):
        return self.__b_z

    @b_z.setter
    def b_z(self, b_z):
        self.__b_z = b_z

    @property
    def delta_W_z(self):
        return self.__delta_W_z

    @delta_W_z.setter
    def delta_W_z(self, delta_W_z):
        self.__delta_W_z = delta_W_z

    @property
    def delta_U_z(self):
        return self.__delta_U_z

    @delta_U_z.setter
    def delta_U_z(self, delta_U_z):
        self.__delta_U_z = delta_U_z

    @property
    def delta_b_z(self):
        return self.__delta_b_z

    @delta_b_z.setter
    def delta_b_z(self, delta_b_z):
        self.__delta_b_z = delta_b_z

    @property
    def W_h(self):
        return self.__W_h

    @W_h.setter
    def W_h(self, W_h):
        self.__W_h = W_h

    @property
    def U_h(self):
        return self.__U_h

    @U_h.setter
    def U_h(self, U_h):
        self.__U_h = U_h

    @property
    def b_h(self):
        return self.__b_h

    @b_h.setter
    def b_h(self, b_h):
        self.__b_h = b_h

    @property
    def delta_W_h(self):
        return self.__delta_W_h

    @delta_W_h.setter
    def delta_W_h(self, delta_W_h):
        self.__delta_W_h = delta_W_h

    @property
    def delta_U_h(self):
        return self.__delta_U_h

    @delta_U_h.setter
    def delta_U_h(self, delta_U_h):
        self.__delta_U_h = delta_U_h

    @property
    def delta_b_h(self):
        return self.__delta_b_h

    @delta_b_h.setter
    def delta_b_h(self, delta_b_h):
        self.__delta_b_h = delta_b_h

    @property
    def params(self):
        if self.use_bias:
            return [self.W_r, self.U_r, self.b_r,
                    self.W_z, self.U_z, self.b_z,
                    self.W_h, self.U_h, self.b_h]
        return [self.W_r, self.U_r,
                self.W_z, self.U_z,
                self.W_h, self.U_h]

    @property
    def grads(self):
        if self.use_bias:
            return [self.delta_W_r, self.delta_U_r, self.delta_b_r,
                    self.delta_W_z, self.delta_U_z, self.delta_b_z,
                    self.delta_W_h, self.delta_U_h, self.delta_b_h]
        return [self.delta_W_r, self.delta_U_r,
                self.delta_W_z, self.delta_U_z,
                self.delta_W_h, self.delta_U_h]

    def call(self, pre_layer=None, *args, **kwargs):
        self.connection(pre_layer)
        return self

    def connection(self, pre_layer):
        super(GRU, self).connection(pre_layer)
        self.W_r = self.kernel_initializer((self.input_dim, self.output_dim))
        self.U_r = self.recurrent_initializer((self.output_dim, self.output_dim))

        self.W_z = self.kernel_initializer((self.input_dim, self.output_dim))
        self.U_z = self.recurrent_initializer((self.output_dim, self.output_dim))

        self.W_h = self.kernel_initializer((self.input_dim, self.output_dim))
        self.U_h = self.recurrent_initializer((self.output_dim, self.output_dim))

        if self.use_bias:
            self.b_r = self.bias_initializer((self.output_dim, ))
            self.b_z = self.bias_initializer((self.output_dim, ))
            self.b_h = self.bias_initializer((self.output_dim, ))

    def forward(self, inputs, *args, **kwargs):
        # inputs: batch_size, time_step, out_dim
        inputs = np.asarray(inputs)
        self.inputs = inputs
        assert list(inputs.shape[1:]) == list(self.input_shape[1:])
        self.input_shape = inputs.shape
        self.output_shape[0] = self.input_shape[0]
        nb_batch, nb_seq, nb_input_dim = self.input_shape

        self.outputs = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.logits_r = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.logits_z = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.logits_h = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.logits_s = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.outputs_r = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.outputs_z = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.outputs_h = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.outputs_s = np.zeros((nb_batch, nb_seq, self.output_dim))
        for t in xrange(nb_seq):
            h_pre = np.zeros((nb_batch, self.output_dim)) \
                        if t == 0 else self.outputs[:, t - 1, :]
            x_now = self.inputs[:,t,:]
            r = x_now.dot(self.W_r) + h_pre.dot(self.U_r)
            z = x_now.dot(self.W_z) + h_pre.dot(self.U_z)
            s = x_now.dot(self.W_h)

            if self.use_bias:
                r += self.b_r
                z += self.b_z
                s += self.b_h

            self.logits_r[:,t,:] = r
            self.logits_z[:,t,:] = z

            r = self.recurrent_activator.forward(r)
            z = self.recurrent_activator.forward(z)

            s = s + (r * h_pre).dot(self.U_h)
            self.logits_s[:,t,:] = s
            s = self.activator.forward(s)

            self.outputs_r[:,t,:] = r
            self.outputs_z[:,t,:] = z
            self.outputs_s[:,t,:] = s

            self.outputs[:,t,:] = z * h_pre + (1 - z) * s

        if self.return_sequences:
            return self.outputs
        else:
            return self.outputs[:,-1,:]

    def backward(self, pre_delta, *args, **kwargs):
        """
        BPTT.

        根据LSTM的BPTT,这个很容易自己推导出来.
        思路步骤:
        1. 求出loss关于h_{t-1},z_t,r_t和s_t的logit（激活函数里面的线性和）的偏导
        2. 所有其余的偏导（W,U,b,以及往前一个timestep和往前一层传的偏导）都可以由上面的几个偏导表示

        # Params
        pre_delta: gradients from the next layer.

        # Return
        gradients to the previous layer.
        """
        self.delta_W_r = np.zeros(self.W_r.shape)
        self.delta_W_z = np.zeros(self.W_z.shape)
        self.delta_W_h = np.zeros(self.W_h.shape)
        self.delta_U_r = np.zeros(self.U_r.shape)
        self.delta_U_z = np.zeros(self.U_z.shape)
        self.delta_U_h = np.zeros(self.U_h.shape)
        if self.use_bias:
            self.delta_b_r = np.zeros(self.b_r.shape)
            self.delta_b_z = np.zeros(self.b_z.shape)
            self.delta_b_h = np.zeros(self.b_h.shape)

        nb_batch, nb_seq, nb_input_dim = self.input_shape
        nb_output_dim = self.output_dim
        self.delta = np.zeros(self.input_shape)
        if self.return_sequences:
            assert len(pre_delta.shape) == 3
            # 此处的time_delta为delta_E/delta_output
            time_delta = pre_delta[:,nb_seq - 1,:]
        else:
            assert len(pre_delta.shape) == 2
            # 此处的time_delta为delta_E/delta_output
            time_delta = pre_delta
        for t in np.arange(nb_seq)[::-1]:
            logit_s = self.logits_s[:,t,:]
            logit_z = self.logits_z[:,t,:]
            logit_r = self.logits_r[:,t,:]

            output_z = self.outputs_z[:,t,:]
            output_r = self.outputs_r[:,t,:]
            output_s = self.outputs_s[:,t,:]

            pre_h = np.zeros((nb_batch, nb_output_dim)) \
                        if t == 0 else self.outputs[:,t - 1,:]
            cur_x = self.inputs[:,t,:]
            # cell state
            pre_delta_s = time_delta * (1 - output_z) * self.activator.backward(logit_s)
            pre_delta_z = (time_delta * pre_h - time_delta * output_s) * \
                          self.recurrent_activator.backward(logit_z)
            pre_delta_r = np.dot(pre_delta_s[:,None,:], self.U_h.T).reshape(-1, nb_output_dim) * \
                          (pre_h * self.recurrent_activator.backward(logit_r))

            # 求U的梯度
            self.delta_U_r += np.dot(pre_h.T, pre_delta_r) / nb_batch
            self.delta_U_z += np.dot(pre_h.T, pre_delta_z) / nb_batch
            self.delta_U_h += np.dot((output_r * pre_h).T, pre_delta_s) / nb_batch
            # 求W的梯度
            self.delta_W_r += np.dot(cur_x.T, pre_delta_r) / nb_batch
            self.delta_W_z += np.dot(cur_x.T, pre_delta_z) / nb_batch
            self.delta_W_h += np.dot(cur_x.T, pre_delta_s) / nb_batch
            # 求b的梯度
            if self.use_bias:
                self.delta_b_r += np.mean(pre_delta_r, axis=0)
                self.delta_b_z += np.mean(pre_delta_z, axis=0)
                self.delta_b_h += np.mean(pre_delta_s, axis=0)
            # 求传到上一层的误差,layerwise
            self.delta[:,t,:] = np.dot(pre_delta_r, self.W_r.T) + \
                                np.dot(pre_delta_z, self.W_z.T) + \
                                np.dot(pre_delta_s, self.W_h.T)
            # 求同一层不同时间的误差,timewise
            if t > 0:
                time_delta = np.dot(pre_delta_z, self.U_z.T) + time_delta * output_z + \
                             np.dot(pre_delta_s[:,None,:], self.U_h.T).reshape(-1, nb_output_dim) * output_r + \
                             np.dot(pre_delta_r, self.U_r.T)
                if self.return_sequences:
                    time_delta += pre_delta[:,t - 1,:]
        return self.delta
