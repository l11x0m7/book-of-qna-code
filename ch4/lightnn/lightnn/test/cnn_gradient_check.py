# -*- encoding:utf-8 -*-
import sys
sys.path.append('../../')

import numpy as np

from lightnn.layers.convolutional import Conv2d
from lightnn.layers.pooling import MaxPoolingLayer, AvgPoolingLayer
from lightnn.base.activations import Sigmoid, Relu, Identity
from lightnn.base.initializers import xavier_uniform_initializer


def conv_gradient_check():
    """
    gradient check for convolution layer
    """
    activator = Relu()
    def init_test():
        a = np.array(
            [[[0,1,1,0,2],
              [2,2,2,2,1],
              [1,0,0,2,0],
              [0,1,1,0,0],
              [1,2,0,0,2]],
             [[1,0,2,2,0],
              [0,0,0,2,0],
              [1,2,1,2,1],
              [1,0,0,0,0],
              [1,2,1,1,1]],
             [[2,1,2,0,0],
              [1,0,0,1,0],
              [0,2,1,0,1],
              [0,1,2,2,2],
              [2,1,0,0,1]]])
        a = a.transpose([1,2,0])
        a = np.expand_dims(a, 0)
        # debug point : when `stride` is `[1, 1]`, the function runs in error states
        cl = Conv2d((3,3), 2, (1,5,5,3), 1, [1, 1], activator=activator,
                    initializer=xavier_uniform_initializer)
        cl.filters[0].weights = np.array(
            [[[-1,1,0],
              [0,1,0],
              [0,1,1]],
             [[-1,-1,0],
              [0,0,0],
              [0,-1,0]],
             [[0,0,-1],
              [0,1,0],
              [1,-1,-1]]], dtype=np.float64).transpose([1,2,0])
        cl.filters[0].b=1
        cl.filters[1].W = np.array(
            [[[1,1,-1],
              [-1,-1,1],
              [0,-1,1]],
             [[0,1,0],
             [-1,0,-1],
              [-1,1,0]],
             [[-1,0,0],
              [-1,0,1],
              [-1,0,0]]], dtype=np.float64).transpose([1,2,0])
        return a, cl

    """
        gradient check
    """

    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o : np.sum(o) / 2
    # 计算forward值
    a, cl = init_test()
    output = cl.forward(a)
    print np.transpose(output, [0, 3, 1, 2])
    # 求取sensitivity map，是一个全1数组
    sensitivity_array =  np.ones(cl.output.shape,
                                dtype=np.float64) / 2
    # 计算梯度
    cl.backward(sensitivity_array)
    # 检查梯度
    epsilon = 1e-4
    for d in range(cl.filters[0].delta_W.shape[0]):
        for i in range(cl.filters[0].delta_W.shape[1]):
            for j in range(cl.filters[0].delta_W.shape[2]):
                cl.filters[0].W[d,i,j] += epsilon
                cl.forward(a)
                err1 = error_function(cl.output)
                cl.filters[0].W[d,i,j] -= 2*epsilon
                cl.forward(a)
                err2 = error_function(cl.output)
                expect_grad = (err1 - err2) / (2 * epsilon)
                cl.filters[0].W[d,i,j] += epsilon
                print 'weights(%d,%d,%d): expected - actural %f - %f' % (
                    d, i, j, expect_grad, cl.filters[0].delta_W[d,i,j])

    cl.filters[0].b += epsilon
    cl.forward(a)

    err1 = error_function(cl.output)
    cl.filters[0].b -= 2*epsilon
    cl.forward(a)
    err2 = error_function(cl.output)
    expect_grad = (err1 - err2) / (2 * epsilon)
    cl.filters[0].b += epsilon
    print 'biases(%d,%d,%d): expected - actural %f - %f' % (
        d, i, j, expect_grad, cl.filters[0].delta_b)


def max_pool_gradient_check():
    """
    gradient check for max pooling layer
    """
    a = np.array(
      [[[0,1,1,0,2],
        [2,2,2,2,1],
        [1,0,0,2,0],
        [0,1,1,0,0],
        [1,2,0,0,2]],
       [[1,0,2,2,0],
        [0,0,0,2,0],
        [1,2,1,2,1],
        [1,0,0,0,0],
        [1,2,1,1,1]],
       [[2,1,2,0,0],
        [1,0,0,1,0],
        [0,2,1,0,1],
        [0,1,2,2,2],
        [2,1,0,0,1]]]).transpose([1,2,0])
    a = np.expand_dims(a, 0)
    mp = MaxPoolingLayer((2,2), (1,5,5,3), [1,1], 0)
    output = mp.forward(a)
    print output.transpose((0,3,1,2))
    sensitivity_array = np.ones(mp.output.shape,
                            dtype=np.float64)
    delta = mp.backward(sensitivity_array)
    print delta.transpose([0,3,1,2])


def avg_pool_gradient_check():
    """
    gradient check for avg pooling layer
    """
    a = np.array(
      [[[0,1,1,0,2],
        [2,2,2,2,1],
        [1,0,0,2,0],
        [0,1,1,0,0],
        [1,2,0,0,2]],
       [[1,0,2,2,0],
        [0,0,0,2,0],
        [1,2,1,2,1],
        [1,0,0,0,0],
        [1,2,1,1,1]],
       [[2,1,2,0,0],
        [1,0,0,1,0],
        [0,2,1,0,1],
        [0,1,2,2,2],
        [2,1,0,0,1]]]).transpose([1,2,0])
    a = np.expand_dims(a, 0)
    mp = AvgPoolingLayer((2,2), (1,5,5,3), [1,1], 0)
    output = mp.forward(a)
    print output.transpose([0,3,1,2])
    sensitivity_array = np.ones(mp.output.shape,
                            dtype=np.float64)
    delta = mp.backward(sensitivity_array)
    print delta.transpose([0,3,1,2])


if __name__ == '__main__':
    conv_gradient_check()
    max_pool_gradient_check()
    avg_pool_gradient_check()
