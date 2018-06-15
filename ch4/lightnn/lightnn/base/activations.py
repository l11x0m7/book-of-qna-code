# -*- encoding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Activator(object):
    def forward(self, z, *args, **kwargs):
        raise NotImplementedError('function `forward` should be implemented')

    def backward(self, z, *args, **kwargs):
        raise NotImplementedError('function `backward` should be implemented')


# --- sigmoid functions ---*

def sigmoid(z):
    z = np.asarray(z)
    return 1. / (1 + np.exp(-z))


def delta_sigmoid(z):
    z = np.asarray(z)
    return sigmoid(z) * (1. - sigmoid(z))


class Sigmoid(Activator):
    def forward(self, z, *args, **kwargs):
        return sigmoid(z)
    def backward(self, z, *args, **kwargs):
        return delta_sigmoid(z)


# --- relu functions ---*

def relu(z, alpha=0., max_value=None):
    z = np.asarray(z)
    x = np.maximum(z, 0)
    if max_value is not None:
        x = np.clip(x, 0, max_value)
    if alpha != 0.:
        negative_part = np.maximum(-z, 0)
        x -= alpha * negative_part
    return x

def delta_relu(z, alpha=0., max_value=None):
    z = np.asarray(z)
    if max_value is not None:
        return (np.where(z <= max_value, z, -1e-6 * z) >= 0).astype(int) \
                        + alpha * (z < 0).astype(int)
    else:
        return (z >= 0).astype(int) + alpha * (z < 0).astype(int)

class Relu(Activator):
    def __init__(self, alpha=0., max_value=None):
        self.alpha = alpha
        self.max_value = max_value
    def forward(self, z, *args, **kwargs):
        return relu(z, self.alpha, self.max_value)
    def backward(self, z, *args, **kwargs):
        return delta_relu(z, self.alpha, self.max_value)


# --- identity functions ---*

def identity(z):
    z = np.asarray(z)
    return z

def delta_identity(z):
    z = np.asarray(z)
    return np.ones(z.shape)

class Identity(Activator):
    def forward(self, z, *args, **kwargs):
        return identity(z)

    def backward(self, z, *args, **kwargs):
        return delta_identity(z)

Linear = Identity


# --- softmax functions ---*

def softmax(z):
    z = np.asarray(z)
    if len(z.shape) > 1:
        z -= z.max(axis=1).reshape([z.shape[0], 1])
        z = np.exp(z)
        z /= np.sum(z, axis=1).reshape([z.shape[0], 1])
        return z
    else:
        z -= np.max(z)
        z = np.exp(z)
        z /= np.sum(z)
        return z

def delta_softmax(z):
    z = np.asarray(z)
    return np.ones(z.shape, dtype=z.dtype)

class Softmax(Activator):
    def forward(self, z, *args, **kwargs):
        return softmax(z)

    def backward(self, z, *args, **kwargs):
        return delta_softmax(z)


# --- tanh functions ---*

def tanh(z):
    z = np.asarray(z)
    return np.tanh(z)

def delta_tanh(z):
    z = np.asarray(z)
    return 1- np.power(np.tanh(z), 2)

class Tanh(Activator):
    def forward(self, z, *args, **kwargs):
        return tanh(z)

    def backward(self, z, *args, **kwargs):
        return delta_tanh(z)


# --- leaky relu functions ---*

def leaky_relu(z, alpha=0.3):
    z = np.asarray(z)
    return np.maximum(z, 0) + np.minimum(z, 0) * alpha

def delta_leaky_relu(z, alpha=0.3):
    z = np.asarray(z)
    return np.greater_equal(z, 0).astype(int) + np.less(z, 0).astype(int) * alpha

class LeakyRelu(Activator):
    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def forward(self, z, *args, **kwargs):
        return leaky_relu(z, self.alpha)

    def backward(self, z, *args, **kwargs):
        return delta_leaky_relu(z, self.alpha)


# --- elu functions ---*

def elu(z, alpha=1.0):
    z = np.asarray(z)
    return np.maximum(z, 0) + alpha * (np.exp(np.minimum(z, 0)) - 1.)

def delta_elu(z, alpha=1.0):
    z = np.asarray(z)
    return np.greater_equal(z, 0).astype(int) + \
                alpha * np.exp(np.minimum(z, 0)) * np.less(z, 0).astype(int)

class Elu(Activator):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, z, *args, **kwargs):
        return elu(z, self.alpha)

    def backward(self, z, *args, **kwargs):
        return delta_elu(z, self.alpha)


# --- selu functions ---*

def selu(z, alpha, scale):
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
    # Arguments
        x: A tensor or variable to compute the activation function for.
    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    z = np.asarray(z)
    return scale * elu(z, alpha)

def delta_selu(z, alpha, scale):
    z = np.asarray(z)
    return scale * delta_elu(z, alpha)

class Selu(Activator):
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def forward(self, z, *args, **kwargs):
        return selu(z, self.alpha, self.scale)

    def backward(self, z, *args, **kwargs):
        return delta_selu(z, self.alpha, self.scale)


# --- thresholded relu functions ---*

def thresholded_relu(z, theta=1.0):
    z = np.asarray(z)
    return z * (np.greater(z, theta).astype(np.float64))

def delta_thresholded_relu(z, theta=1.0):
    z = np.asarray(z)
    return np.greater(z, theta).astype(np.float64)

class ThresholdedReLU(Activator):
    """Thresholded Rectified Linear Unit.

    It follows:
    `f(x) = x for x > theta`,
    `f(x) = 0 otherwise`.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        theta: float >= 0. Threshold location of activation.

    # References
        - [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/abs/1402.3337)
    """
    def __init__(self, theta=1.0):
        self.theta = theta

    def forward(self, z, *args, **kwargs):
        return thresholded_relu(z, self.theta)

    def backward(self, z, *args, **kwargs):
        return delta_thresholded_relu(z, self.theta)


# --- softplus functions ---*

def softplus(z):
    z = np.asarray(z)
    return np.log(1 + np.exp(z))

def delta_softplus(z):
    z = np.asarray(z)
    return np.exp(z) / (1 + np.exp(z))

class Softplus(Activator):
    def forward(self, z, *args, **kwargs):
        return softplus(z)

    def backward(self, z, *args, **kwargs):
        return delta_softplus(z)


# --- softsign functions ---*

def softsign(z):
    z = np.asarray(z)
    return z / (np.abs(z) + 1)

def delta_softsign(z):
    z = np.asarray(z)
    return 1. / np.power(np.abs(z) + 1, 2)

class Softsign(Activator):
    def forward(self, z, *args, **kwargs):
        return softsign(z)

    def backward(self, z, *args, **kwargs):
        return delta_softsign(z)


# --- hard sigmoid functions ---*

def hard_sigmoid(z):
    """Segment-wise linear approximation of sigmoid.

    Faster than sigmoid.
    Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
    In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

    # Arguments
        z: A tensor or variable.

    # Returns
        A tensor.
    """
    z = np.asarray(z)
    z = (0.2 * z) + 0.5
    z = np.clip(z, 0., 1.)
    return z

def delta_hard_sigmoid(z):
    z = np.asarray(z)
    z = (0.2 * z) + 0.5
    return (np.where(z <= 1, z, -1e-6 * z) >= 0.).astype(int) * 0.2

class HardSigmoid(Activator):
    def forward(self, z, *args, **kwargs):
        return hard_sigmoid(z)

    def backward(self, z, *args, **kwargs):
        return delta_hard_sigmoid(z)


def get(activator):
    if activator is None:
        return Identity()
    if isinstance(activator, str):
        activator = activator.lower()
        if activator in ('linear', 'identity'):
            return Identity()
        elif activator in ('sigmoid', ):
            return Sigmoid()
        elif activator in ('relu', ):
            return Relu()
        elif activator in ('softmax', ):
            return Softmax()
        elif activator in ('tanh', ):
            return Tanh()
        elif activator in ('leaky_relu', 'leakyrelu'):
            return LeakyRelu()
        elif activator in ('elu', ):
            return Elu()
        elif activator in ('selu', ):
            return Selu()
        elif activator in ('thresholded_relu', 'thresholdedrelu'):
            return ThresholdedReLU()
        elif activator in ('softplus', ):
            return Softplus()
        elif activator in ('softsign', ):
            return Softsign()
        elif activator in ('hard_sigmoid', 'hardsigmoid'):
            return HardSigmoid()
        else:
            raise ValueError('Unknown activator name `{}`'.format(activator))
    elif isinstance(activator, Activator):
        return activator
    else:
        raise ValueError('Unknown activator type `{}`'.format(activator.__class__.__name__))
