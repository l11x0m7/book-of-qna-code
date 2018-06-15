# -*- encoding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def xavier_uniform_initializer(shape):
    """Defines an initializer for the Xavier distribution.

    This function will be used as a variable scope initializer.

    Args:
    :param shape: Tuple or 1-d array that species dimensions of requested tensor.
    :return: specified shape sampled from Xavier distribution.
    """
    m = shape[0]
    n = shape[1] if len(shape)>1 else shape[0]
    bound = np.sqrt(6. / (m + n))
    out = np.random.uniform(-bound, bound, shape)
    return out


glorot_uniform_initializer = xavier_uniform_initializer


def default_weight_initializer(shape):
    """Initialize weights with random distribution

    :param shape: Tuple or 1-d array that species dimensions of requested tensor.
    :return: specified shape sampled from small scale random distribution.
    """
    return np.random.randn(shape) / np.sqrt(shape[1])


def large_weight_initializer(shape):
    """Initialize weights with large scale random distribution
    Not quite recommended to use.

    # Arguments

    shape: Tuple or 1-d array that species dimensions of requested tensor.

    # Return

    specified shape sampled from large scale random distribution.
    """
    return np.random.randn(shape)


def orthogonal_initializer(shape, gain=1., seed=None):
    """Initializer that generates a random orthogonal matrix.

    # Arguments

        gain: Multiplicative factor to apply to the orthogonal matrix.
        seed: A Python integer. Used to seed the random generator.

    # References

        Saxe et al., http://arxiv.org/abs/1312.6120
    """
    num_rows = 1
    for dim in shape[:-1]:
        num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (num_rows, num_cols)
    if seed is not None:
        np.random.seed(seed)
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # Pick the one with the correct shape.
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return gain * q[:shape[0], :shape[1]]


def zeros(shape):
    return np.zeros(shape)


def ones(shape):
    return np.ones(shape)


def get(initializer):
    if isinstance(initializer, str):
        initializer = initializer.lower()
        if initializer in ('glorot_uniform_initializer', 'glorot uniform initializer',
                           'xavier_uniform_initializer', 'xavier uniform initializer'):
            return xavier_uniform_initializer
        elif initializer in ('default_weight_initializer', 'default weight initializer'):
            return default_weight_initializer
        elif initializer in ('large_weight_initializer', 'large weight initializer'):
            return large_weight_initializer
        elif initializer in ('orthogonal_initializer', 'orthogonal initializer'):
            return orthogonal_initializer
        elif initializer in ('zeros', 'zero'):
            return zeros
        elif initializer in ('ones', 'one'):
            return ones
        else:
            raise ValueError('Unknown initializer name `{}`'.format(initializer))
    elif isinstance(initializer, type(lambda k:k)):
        return initializer
    else:
        raise ValueError('Unknown initializer type `{}`'.format(initializer.__name__))
