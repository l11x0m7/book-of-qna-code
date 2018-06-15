#-*- encoding:utf-8 -*-

import numpy as np

from lightnn.base.activations import *


def test_relu():
    a = np.random.uniform(-5, 5, [3,3])
    relu = Relu(0.5, 2)
    print(a)
    print(relu.forward(a))
    print(relu.backward(a))


if __name__ == '__main__':
    test_relu()