# -*- encoding:utf-8 -*-
import numpy as np

from lightnn.layers.recurrent import SimpleRNN, LSTM, GRU


def vector_rnn_gradient_check():
    batch_size = 1
    time_step = 10
    out_dim = 15
    in_dim = 20
    use_bias = True

    data = np.random.random((batch_size, time_step, in_dim))

    sr = SimpleRNN(out_dim, (batch_size, time_step, in_dim), activator='leaky_relu', use_bias=use_bias)

    loss = lambda p: np.sum(p)
    pre_delta = np.ones((batch_size, out_dim))

    # check real grad
    y_hat = sr.forward(data)
    sr.backward(pre_delta)

    epsilon = 1e-5

    # check W
    for i in xrange(in_dim):
        for j in xrange(out_dim):
            bp_grad = sr.delta_W[i,j]
            sr.W[i,j] -= epsilon
            y_h1 = sr.forward(data)
            loss1 = loss(y_h1)
            sr.W[i,j] += 2 * epsilon
            y_h2 = sr.forward(data)
            loss2 = loss(y_h2)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('W[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

    # check U
    for i in xrange(out_dim):
        for j in xrange(out_dim):
            bp_grad = sr.delta_U[i,j]
            sr.U[i,j] -= epsilon
            y_h1 = sr.forward(data)
            loss1 = loss(y_h1)
            sr.U[i,j] += 2 * epsilon
            y_h2 = sr.forward(data)
            loss2 = loss(y_h2)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('U[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

    # check b
    if use_bias:
        for i in xrange(out_dim):
            bp_grad = sr.delta_b[i]
            sr.b[i] -= epsilon
            y_h1 = sr.forward(data)
            loss1 = loss(y_h1)
            sr.b[i] += 2 * epsilon
            y_h2 = sr.forward(data)
            loss2 = loss(y_h2)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('b[{}],real grad:{}, bp grad:{}'.format(i, real_grad, bp_grad))


def sequence_rnn_gradient_check():
    batch_size = 10
    time_step = 10
    out_dim = 15
    out_dim2 = 5
    in_dim = 20
    use_bias = True

    data = np.random.random((batch_size, time_step, in_dim))

    sr1 = SimpleRNN(out_dim, (batch_size, time_step, in_dim),
                    activator='leaky_relu', use_bias=use_bias, return_sequences=True)
    sr2 = SimpleRNN(out_dim2)(sr1)

    loss = lambda p: np.sum(p)
    pre_delta = np.ones((batch_size, out_dim2))

    # check real grad
    y1_hat = sr1.forward(data)
    y2_hat = sr2.forward(y1_hat)
    delta2 = sr2.backward(pre_delta)
    delta1 = sr1.backward(delta2)

    epsilon = 1e-5

    # check W
    for i in xrange(in_dim):
        for j in xrange(out_dim):
            bp_grad = sr1.delta_W[i,j]
            sr1.W[i,j] -= epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss1 = loss(y2_hat)
            sr1.W[i,j] += 2 * epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss2 = loss(y2_hat)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('W[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

    # check U
    for i in xrange(out_dim):
        for j in xrange(out_dim):
            bp_grad = sr1.delta_U[i,j]
            sr1.U[i,j] -= epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss1 = loss(y2_hat)
            sr1.U[i,j] += 2 * epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss2 = loss(y2_hat)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('U[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

    # check b
    if use_bias:
        for i in xrange(out_dim):
            bp_grad = sr1.delta_b[i]
            sr1.b[i] -= epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss1 = loss(y2_hat)
            sr1.b[i] += 2 * epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss2 = loss(y2_hat)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('b[{}],real grad:{}, bp grad:{}'.format(i, real_grad, bp_grad))


def vector_lstm_gradient_check():
    batch_size = 10
    time_step = 10
    out_dim = 15
    in_dim = 20
    use_bias = True

    data = np.random.random((batch_size, time_step, in_dim))

    sr = LSTM(out_dim, (batch_size, time_step, in_dim), activator='tanh', use_bias=use_bias)

    loss = lambda p: np.sum(p)
    pre_delta = np.ones((batch_size, out_dim))

    # check real grad
    y_hat = sr.forward(data)
    sr.backward(pre_delta)

    epsilon = 1e-8

    # check W_i
    for i in xrange(in_dim):
        for j in xrange(out_dim):
            bp_grad = sr.delta_W_i[i,j]
            sr.W_i[i,j] -= epsilon
            y_h1 = sr.forward(data)
            loss1 = loss(y_h1)
            sr.W_i[i,j] += 2 * epsilon
            y_h2 = sr.forward(data)
            loss2 = loss(y_h2)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('W_i[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

    # check U_o
    for i in xrange(out_dim):
        for j in xrange(out_dim):
            bp_grad = sr.delta_U_o[i,j]
            sr.U_o[i,j] -= epsilon
            y_h1 = sr.forward(data)
            loss1 = loss(y_h1)
            sr.U_o[i,j] += 2 * epsilon
            y_h2 = sr.forward(data)
            loss2 = loss(y_h2)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('U_o[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

    # check b_f
    if use_bias:
        for i in xrange(out_dim):
            bp_grad = sr.delta_b_f[i]
            sr.b_f[i] -= epsilon
            y_h1 = sr.forward(data)
            loss1 = loss(y_h1)
            sr.b_f[i] += 2 * epsilon
            y_h2 = sr.forward(data)
            loss2 = loss(y_h2)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('b_f[{}],real grad:{}, bp grad:{}'.format(i, real_grad, bp_grad))

     # check b_c
    if use_bias:
        for i in xrange(out_dim):
            bp_grad = sr.delta_b_c[i]
            sr.b_c[i] -= epsilon
            y_h1 = sr.forward(data)
            loss1 = loss(y_h1)
            sr.b_c[i] += 2 * epsilon
            y_h2 = sr.forward(data)
            loss2 = loss(y_h2)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('b_c[{}],real grad:{}, bp grad:{}'.format(i, real_grad, bp_grad))


def sequence_lstm_gradient_check():
    batch_size = 10
    time_step = 10
    out_dim = 15
    out_dim2 = 5
    in_dim = 20
    use_bias = True

    data = np.random.random((batch_size, time_step, in_dim))

    sr1 = LSTM(out_dim, (batch_size, time_step, in_dim),
                    activator='tanh', use_bias=use_bias, return_sequences=True)
    sr2 = LSTM(out_dim2)(sr1)

    loss = lambda p: np.sum(p)
    pre_delta = np.ones((batch_size, out_dim2))

    # check real grad
    y1_hat = sr1.forward(data)
    y2_hat = sr2.forward(y1_hat)
    delta2 = sr2.backward(pre_delta)
    delta1 = sr1.backward(delta2)

    epsilon = 1e-5

    # check W_i
    for i in xrange(in_dim):
        for j in xrange(out_dim):
            bp_grad = sr1.delta_W_i[i,j]
            sr1.W_i[i,j] -= epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss1 = loss(y2_hat)
            sr1.W_i[i,j] += 2 * epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss2 = loss(y2_hat)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('W_i[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

    # check U_o
    for i in xrange(out_dim):
        for j in xrange(out_dim):
            bp_grad = sr1.delta_U_o[i,j]
            sr1.U_o[i,j] -= epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss1 = loss(y2_hat)
            sr1.U_o[i,j] += 2 * epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss2 = loss(y2_hat)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('U_o[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

    # check b_f
    if use_bias:
        for i in xrange(out_dim):
            bp_grad = sr1.delta_b_f[i]
            sr1.b_f[i] -= epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss1 = loss(y2_hat)
            sr1.b_f[i] += 2 * epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss2 = loss(y2_hat)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('b_f[{}],real grad:{}, bp grad:{}'.format(i, real_grad, bp_grad))

    # check b_c
    if use_bias:
        for i in xrange(out_dim):
            bp_grad = sr1.delta_b_c[i]
            sr1.b_c[i] -= epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss1 = loss(y2_hat)
            sr1.b_c[i] += 2 * epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss2 = loss(y2_hat)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('b_c[{}],real grad:{}, bp grad:{}'.format(i, real_grad, bp_grad))


def vector_gru_gradient_check():
    batch_size = 10
    time_step = 20
    out_dim = 15
    in_dim = 20
    use_bias = True

    data = np.random.random((batch_size, time_step, in_dim))

    sr = GRU(out_dim, (batch_size, time_step, in_dim), activator='tanh', use_bias=use_bias)

    loss = lambda p: np.sum(p)
    pre_delta = np.ones((batch_size, out_dim))

    # check real grad
    y_hat = sr.forward(data)
    sr.backward(pre_delta)

    epsilon = 1e-8

    # check W_r
    for i in xrange(in_dim):
        for j in xrange(out_dim):
            bp_grad = sr.delta_W_r[i,j]
            sr.W_r[i,j] -= epsilon
            y_h1 = sr.forward(data)
            loss1 = loss(y_h1)
            sr.W_r[i,j] += 2 * epsilon
            y_h2 = sr.forward(data)
            loss2 = loss(y_h2)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('W_r[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

    # check U_z
    for i in xrange(out_dim):
        for j in xrange(out_dim):
            bp_grad = sr.delta_U_z[i,j]
            sr.U_z[i,j] -= epsilon
            y_h1 = sr.forward(data)
            loss1 = loss(y_h1)
            sr.U_z[i,j] += 2 * epsilon
            y_h2 = sr.forward(data)
            loss2 = loss(y_h2)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('U_z[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

    # check b_h
    if use_bias:
        for i in xrange(out_dim):
            bp_grad = sr.delta_b_h[i]
            sr.b_h[i] -= epsilon
            y_h1 = sr.forward(data)
            loss1 = loss(y_h1)
            sr.b_h[i] += 2 * epsilon
            y_h2 = sr.forward(data)
            loss2 = loss(y_h2)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('b_h[{}],real grad:{}, bp grad:{}'.format(i, real_grad, bp_grad))

     # check b_z
    if use_bias:
        for i in xrange(out_dim):
            bp_grad = sr.delta_b_z[i]
            sr.b_z[i] -= epsilon
            y_h1 = sr.forward(data)
            loss1 = loss(y_h1)
            sr.b_z[i] += 2 * epsilon
            y_h2 = sr.forward(data)
            loss2 = loss(y_h2)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('b_z[{}],real grad:{}, bp grad:{}'.format(i, real_grad, bp_grad))


def sequence_gru_gradient_check():
    batch_size = 10
    time_step = 10
    out_dim = 15
    out_dim2 = 5
    in_dim = 20
    use_bias = True

    data = np.random.random((batch_size, time_step, in_dim))

    sr1 = GRU(out_dim, (batch_size, time_step, in_dim),
                    activator='tanh', use_bias=use_bias, return_sequences=True)
    sr2 = GRU(out_dim2)(sr1)

    loss = lambda p: np.sum(p)
    pre_delta = np.ones((batch_size, out_dim2))

    # check real grad
    y1_hat = sr1.forward(data)
    y2_hat = sr2.forward(y1_hat)
    delta2 = sr2.backward(pre_delta)
    delta1 = sr1.backward(delta2)

    epsilon = 1e-5

    # check W_r
    for i in xrange(in_dim):
        for j in xrange(out_dim):
            bp_grad = sr1.delta_W_r[i,j]
            sr1.W_r[i,j] -= epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss1 = loss(y2_hat)
            sr1.W_r[i,j] += 2 * epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss2 = loss(y2_hat)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('W_r[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

    # check U_z
    for i in xrange(out_dim):
        for j in xrange(out_dim):
            bp_grad = sr1.delta_U_z[i,j]
            sr1.U_z[i,j] -= epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss1 = loss(y2_hat)
            sr1.U_z[i,j] += 2 * epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss2 = loss(y2_hat)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('U_z[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

    # check b_h
    if use_bias:
        for i in xrange(out_dim):
            bp_grad = sr1.delta_b_h[i]
            sr1.b_h[i] -= epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss1 = loss(y2_hat)
            sr1.b_h[i] += 2 * epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss2 = loss(y2_hat)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('b_h[{}],real grad:{}, bp grad:{}'.format(i, real_grad, bp_grad))

    # check b_z
    if use_bias:
        for i in xrange(out_dim):
            bp_grad = sr1.delta_b_z[i]
            sr1.b_z[i] -= epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss1 = loss(y2_hat)
            sr1.b_z[i] += 2 * epsilon
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss2 = loss(y2_hat)
            real_grad = (loss2 - loss1) / (2 * epsilon) / batch_size
            print('b_z[{}],real grad:{}, bp grad:{}'.format(i, real_grad, bp_grad))


if __name__ == '__main__':
    # vector_rnn_gradient_check()
    # sequence_rnn_gradient_check()
    # vector_lstm_gradient_check()
    # sequence_lstm_gradient_check()
    # vector_gru_gradient_check()
    sequence_gru_gradient_check()
