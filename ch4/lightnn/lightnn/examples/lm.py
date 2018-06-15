# -*- coding: utf-8 -*-

import os

import numpy as np

from lightnn.layers import AvgPooling, MaxPooling
from lightnn.layers import SimpleRNN, LSTM, GRU, Softmax, Flatten
from lightnn.models import Sequential
from lightnn.base import Adam


def get_data():
    corpus_path = os.path.join(os.path.dirname(__file__), 'data/tiny_shakespeare.txt')
    raw_text = open(corpus_path, 'r').read()
    chars = list(set(raw_text))
    data_size, vocab_size = len(raw_text), len(chars)
    print("data has %s charactres, %s unique." % (data_size, vocab_size))
    char_to_index = {ch: i for i, ch in enumerate(chars)}

    time_steps, batch_size = 20, 50

    length = batch_size * 2000
    text_pointers = np.random.randint(data_size - time_steps - 1, size=length)
    batch_in = np.zeros([length, time_steps, vocab_size])
    batch_out = np.zeros([length, vocab_size], dtype=np.uint8)
    for i in range(length):
        b_ = [char_to_index[c] for c in raw_text[text_pointers[i]:text_pointers[i] + time_steps + 1]]
        batch_in[i, range(time_steps), b_[:-1]] = 1
        batch_out[i, b_[-1]] = 1

    return batch_size, vocab_size, time_steps, batch_in, batch_out


def main1(max_iter):
    batch_size, vocab_size, time_steps, batch_in, batch_out = get_data()

    print("Building model ...")
    net = Sequential()
    net.add(SimpleRNN(output_dim=100, input_shape=(batch_size, time_steps, vocab_size),
                      return_sequences=True))
    net.add(SimpleRNN(output_dim=100, return_sequences=True))
    net.add(MaxPooling(window_shape=(time_steps, 1)))
    net.add(Flatten())
    net.add(Softmax(output_dim=vocab_size))

    net.compile(loss='CCE', optimizer=Adam(lr=0.01, grad_clip=5.))

    print("Train model ...")
    net.fit(batch_in, batch_out, epochs=max_iter, batch_size=batch_size, verbose=2)


def main2(max_iter):
    batch_size, vocab_size, time_steps, batch_in, batch_out = get_data()

    print("Building model ...")
    net = Sequential()
    net.add(SimpleRNN(output_dim=100, input_shape=(batch_size, time_steps, vocab_size),
                      return_sequences=True))
    net.add(SimpleRNN(output_dim=100))
    net.add(Softmax(output_dim=vocab_size))

    net.compile(loss='CCE', optimizer=Adam(lr=0.001, grad_clip=5.))

    print("Train model ...")
    net.fit(batch_in, batch_out, epochs=max_iter, batch_size=batch_size, verbose=2)


def main3(max_iter):
    batch_size, vocab_size, time_steps, batch_in, batch_out = get_data()

    print("Building model ...")
    net = Sequential()
    net.add(LSTM(output_dim=100, input_shape=(batch_size, time_steps, vocab_size),
                      return_sequences=True))
    net.add(LSTM(output_dim=100))
    net.add(Softmax(output_dim=vocab_size))

    net.compile(loss='CCE', optimizer=Adam(lr=0.001, grad_clip=5.))

    print("Train model ...")
    net.fit(batch_in, batch_out, epochs=max_iter, batch_size=batch_size, verbose=2)


def main4(max_iter):
    batch_size, vocab_size, time_steps, batch_in, batch_out = get_data()

    print("Building model ...")
    net = Sequential()
    net.add(GRU(output_dim=100, input_shape=(batch_size, time_steps, vocab_size),
                      return_sequences=True))
    net.add(GRU(output_dim=100))
    net.add(Softmax(output_dim=vocab_size))

    net.compile(loss='CCE', optimizer=Adam(lr=0.001, grad_clip=5.))

    print("Train model ...")
    net.fit(batch_in, batch_out, epochs=max_iter, batch_size=batch_size, verbose=2)


if __name__ == '__main__':
    main2(100)
