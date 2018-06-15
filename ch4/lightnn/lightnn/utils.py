# -*- encoding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    # Args:
    inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.

    # Returns:
    A label smoothing Tensor with the same shape as `inputs`.

    # Examples
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    inputs = np.asarray(inputs)
    K = inputs.shape[-1] # number of channels
    return (1 - epsilon) * inputs + epsilon / K


def sparse_to_dense(index, output_shape, values):
    '''Transform a sparse matrix to a dense matrix

    ```python
    # If index is scalar
    outputs[index] = values

    # If index is a vector, then for each i
    outputs[index[i]] = values[i]

    # If index is an n by d matrix, then for each i in [0, n)
    outputs[index[i][0], ..., index[i][d-1]] = values[i]
    ```

    Indices should be sorted in lexicographic order, and indices must not
    contain any repeats. If `validate_indices` is True, these properties
    are checked during execution.

    Args:
    index: A 0-D, 1-D, or 2-D `Tensor` of type `int32` or `int64`.
    `index[i]` contains the complete index where `values[i]`
    will be placed.
    output_shape: A 1-D `Tensor` of the same type as `index`.  Shape
    of the dense output tensor.
    values: A 0-D or 1-D `Tensor`.  Values corresponding to each row of
    `index`, or a scalar value to be used for all sparse indices.

    Returns:
    Dense `Tensor` of shape `output_shape`.  Has the same type as
    `values`.

    # Examples
    ```
    index = np.asarray([0,1,0,1,0]).reshape((-1, 1))
    index = np.concatenate([np.arange(0, 5).reshape((-1, 1)), index], axis=1)
    output_shape = [5, 2]
    output_values = np.ones((5, ))
    print(sparse_to_dense(index, output_shape, output_values))

    >>
    [[ 1.  0.]
    [ 0.  1.]
    [ 1.  0.]
    [ 0.  1.]
    [ 1.  0.]]
    ```
    '''
    outputs = np.zeros(output_shape)
    index = np.asarray(index)
    if np.isscalar(index):
        outputs[index] = values
    elif isinstance(index, np.ndarray):
        for i, ind in enumerate(index):
            if np.isscalar(ind):
                outputs[ind] = values[i]
            else:
                outputs[tuple(ind.tolist())] = 1
    return outputs
