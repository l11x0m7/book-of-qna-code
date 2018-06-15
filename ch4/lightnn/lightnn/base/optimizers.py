# -*- encoding:utf-8 -*-

import numpy as np


class Optimizer(object):
    def __init__(self, lr=1e-3, decay=0., grad_clip=-1, lr_min=0., lr_max=np.inf):
        self.lr = lr
        self.decay = decay
        self.clip = grad_clip
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.iterations = 0

    def update(self):
        self.iterations += 1
        self.lr *= (1. / 1 + self.decay * self.iterations)
        self.lr = np.clip(self.lr, self.lr_min, self.lr_max)


class SGD(Optimizer):
    def __init__(self, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)

    def minimize(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * _grad_clip(g, self.clip)
        super(SGD, self).update()

    def maximum(self, params, grads):
        for p, g in zip(params, grads):
            p += self.lr * _grad_clip(g, self.clip)
        super(SGD, self).update()


class Momentum(Optimizer):
    """
    Performs stochastic gradient descent with momentum.

    momentum: Scalar between 0 and 1 giving the momentum value.
              Setting momentum = 0 reduces to sgd.
    velocity: A numpy array of the same shape as w and dw used to store a moving
              average of the gradients.
    """
    def __init__(self, momentum=0.9, nesterov=False, *args, **kwargs):
        super(Momentum, self).__init__(*args, **kwargs)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = dict()

    def minimize(self, params, grads):
        for p, g in zip(params, grads):
            v = self.velocity.get(id(p), np.zeros_like(p))
            v = self.momentum * v - self.lr * g
            if self.nesterov:
                p = p + self.momentum * v - self.lr * g
            else:
                p = p + v
            self.velocity[id(p)] = v
        super(Momentum, self).update()

    def maximum(self, params, grads):
        for p, g in zip(params, grads):
            v = self.velocity.get(id(p), np.zeros_like(p))
            v = self.momentum * v - self.lr * g
            p -= v
            self.velocity[id(p)] = p
        super(Momentum, self).update()


class RMSProp(Optimizer):
    """
    Uses the RMSProp update rule, which uses a moving average of squared gradient
    values to set adaptive per-parameter learning rates.

    learning_rate: Scalar learning rate.
    decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
        gradient cache.
    epsilon: Small scalar used for smoothing to avoid dividing by zero.
    cache: Moving average of second moments of gradients.
    """
    def __init__(self, decay_rate=0.99, epsilon=1e-8, *args, **kwargs):
        super(RMSProp, self).__init__(*args, **kwargs)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = dict()

    def minimize(self, params, grads):
        for p, g in zip(params, grads):
            cache = self.cache.get(id(p), np.zeros_like(p))
            self.cache[id(p)] = self.decay_rate * cache + (1 - self.decay_rate) * (g ** 2)
            p -= self.lr * g / (np.sqrt(self.cache[id(p)]) + self.epsilon)
        super(RMSProp, self).update()

    def maximum(self, params, grads):
        for p, g in zip(params, grads):
            cache = self.cache.get(id(p), np.zeros_like(p))
            self.cache[id(p)] = self.decay_rate * cache + (1 - self.decay_rate) * (g ** 2)
            p += self.lr * g / (np.sqrt(self.cache[id(p)]) + self.epsilon)
        super(RMSProp, self).update()


class Adam(Optimizer):
    """
        Uses the Adam update rule, which incorporates moving averages of both the
        gradient and its square and a bias correction term.

        beta1: Decay rate for moving average of first moment of gradient.
        beta2: Decay rate for moving average of second moment of gradient.
        epsilon: Small scalar used for smoothing to avoid dividing by zero.
        m: Moving average of gradient.
        v: Moving average of squared gradient.
    """
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, *args, **kwargs):
        super(Adam, self).__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = dict()
        self.v = dict()

    def minimize(self, params, grads):
        for p, g in zip(params, grads):
            m = self.m.get(id(p), np.zeros_like(p))
            v = self.v.get(id(p), np.zeros_like(p))
            self.m[id(p)] = self.beta1 * m + (1 - self.beta1) * g
            self.v[id(p)] = self.beta2 * v + (1 - self.beta2) * g ** 2
            mb = self.m[id(p)] / (1 - self.beta1 ** (self.iterations + 1))
            vb = self.v[id(p)] / (1 - self.beta2 ** (self.iterations + 1))
            p -= (self.lr * mb / (np.sqrt(vb) + self.epsilon))
        super(Adam, self).update()

    def maximum(self, params, grads):
        for p, g in zip(params, grads):
            m = self.m.get(id(p), np.zeros_like(p))
            v = self.v.get(id(p), np.zeros_like(p))
            self.m[id(p)] = self.beta1 * m + (1 - self.beta1) * g
            self.v[id(p)] = self.beta2 * v + (1 - self.beta2) * g
            mb = self.m[id(p)] / (1 - self.beta1 ** (self.iterations + 1))
            vb = self.v[id(p)] / (1 - self.beta2 ** (self.iterations + 1))
            p += (self.lr * mb / (np.sqrt(vb) + self.epsilon))
        super(Adam, self).update()


class Adagrad(Optimizer):
    def __init__(self, epsilon=1e-7, *args, **kwargs):
        super(Adagrad, self).__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.__r = dict()

    def minimize(self, params, grads):
        for p, g in zip(params, grads):
            self.__r.setdefault(id(p), np.zeros_like(p))
            self.__r[id(p)] += g ** 2
            p -= self.lr / (self.epsilon + np.sqrt(self.__r[id(p)])) * g
        super(Adagrad, self).update()

    def maximum(self, params, grads):
        for p, g in zip(params, grads):
            self.__r.setdefault(id(p), np.zeros_like(p))
            self.__r[id(p)] += g ** 2
            p += self.lr / (self.epsilon + np.sqrt(self.__r[id(p)])) * g
        super(Adagrad, self).update()


class Adadelta(Optimizer):
    """Adadelta optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Learning rate.
            It is recommended to leave it at the default value.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [Adadelta - an adaptive learning rate method](http://arxiv.org/abs/1212.5701)
    """

    def __init__(self, lr=1.0, rho=0.95, epsilon=1e-8, decay=0.,
                 *args, **kwargs):
        super(Adadelta, self).__init__(lr=lr, decay=decay, *args, **kwargs)
        self.rho = rho
        self.epsilon = epsilon
        self.__accmulators = dict()
        self.__delta_accumulators = dict()

    def minimize(self, params, grads):
        shapes = [p.shape for p in params]
        # accumulate gradients
        accumulators = [np.zeros(shape) for shape in shapes]
        # accumulate updates
        delta_accumulators = [np.zeros(shape) for shape in shapes]
        self.weights = accumulators + delta_accumulators
        self.updates = []

        for p, g in zip(params, grads):
            a = self.__accmulators.setdefault(id(p), np.zeros_like(p))
            d_a = self.__delta_accumulators.setdefault(id(p), np.zeros_like(p))
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * np.square(g)

            # use the new accumulator and the *old* delta_accumulator
            update = g * np.sqrt(d_a + self.epsilon) / np.sqrt(new_a + self.epsilon)

            p -= self.lr * update

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * np.square(update)
            self.__accmulators[id(p)] = new_a
            self.__delta_accumulators[id(p)] = new_d_a
        super(Adadelta, self).update()


def _grad_clip(grad, clip):
    if clip > 0:
        return np.clip(grad, -clip, clip)
    else:
        return grad


# name aliases
sgd = SGD
momentum = Momentum
rmsprop = RMSProp
adam = Adam
adagrad = Adagrad
adadelta = Adadelta


def get(optimizer):
    if isinstance(optimizer, str):
        optimizer = optimizer.lower()
        if optimizer in ('sgd', ):
            return SGD()
        elif optimizer in ('momentum', ):
            return Momentum()
        elif optimizer in ('rmsprop', 'rms'):
            return RMSProp()
        elif optimizer in ('adam'):
            return Adam()
        elif optimizer in ('adagrad', ):
            return Adagrad()
        elif optimizer in ('adadelta',):
            return Adadelta()
        else:
            raise ValueError('Unknown optimizer name `{}`'.format(optimizer))
    elif isinstance(optimizer, Optimizer):
        return optimizer
    else:
        raise ValueError('Unknown optimizer type `{}`'.format(optimizer.__class__.__name__))
