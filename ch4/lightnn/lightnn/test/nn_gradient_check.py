# -*- encoding:utf-8 -*-

import numpy as np

from lightnn.layers import Dense


batch_size = 1
out_dim = 15
in_dim = 20

data = np.random.random((batch_size, in_dim))

d1 = Dense(out_dim, in_dim, activator='leaky_relu')
d2 = Dense(out_dim)(d1)

pre_delta = np.ones((batch_size, out_dim))
loss = lambda p: np.sum(p)

# check real grad
d1_out = d1.forward(data)
d2_out = d2.forward(d1_out)
d2_bp_out = d2.backward(pre_delta)
d1_bp_out = d1.backward(d2_bp_out)

epsilon = 1e-5

# check W of layer1
for i in xrange(out_dim):
    for j in xrange(in_dim):
        bp_grad = d1.delta_W[i,j]
        d1.W[i,j] -= epsilon
        d1_out = d1.forward(data)
        d2_out = d2.forward(d1_out)
        loss1 = loss(d2_out)
        d1.W[i,j] += 2 * epsilon
        d1_out = d1.forward(data)
        d2_out = d2.forward(d1_out)
        loss2 = loss(d2_out)
        real_grad = (loss2 - loss1) / (2 * epsilon)
        print('W[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

# check b of layer1
for i in xrange(out_dim):
    bp_grad = d1.delta_b[i]
    d1.b[i] -= epsilon
    d1_out = d1.forward(data)
    d2_out = d2.forward(d1_out)
    loss1 = loss(d2_out)
    d1.b[i] += 2 * epsilon
    d1_out = d1.forward(data)
    d2_out = d2.forward(d1_out)
    loss2 = loss(d2_out)
    real_grad = (loss2 - loss1) / (2 * epsilon)
    print('b[{}],real grad:{}, bp grad:{}'.format(i, real_grad, bp_grad))
