# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 Hai Liang Wang<hailiang.hl.wang@gmail.com> All Rights Reserved
#
#
# File: /Users/hain/ai/InsuranceQA-Machine-Learning/deep_qa_1/network.py
# Author: Hai Liang Wang
# Date: 2017-08-08:18:32:05
#
#===============================================================================

"""
   A Simple Network to learning QA.


"""
from __future__ import print_function
from __future__ import division

__copyright__ = "Copyright (c) 2017 Hai Liang Wang. All Rights Reserved"
__author__    = "Hai Liang Wang"
__date__      = "2017-08-08:18:32:05"


import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"

import random
import insuranceqa_data as insuranceqa

data_dir = './insurance_data'

_train_data = insuranceqa.load_pairs_train(data_dir)
_test_data = insuranceqa.load_pairs_test(data_dir)
_valid_data = insuranceqa.load_pairs_valid(data_dir)


'''
build vocab data with more placeholder
'''
vocab_data = insuranceqa.load_pairs_vocab(data_dir)
print("keys", vocab_data.keys())
vocab_size = len(vocab_data['word2id'].keys())
VOCAB_PAD_ID = vocab_size+1
VOCAB_GO_ID = vocab_size+2
vocab_data['word2id']['<PAD>'] = VOCAB_PAD_ID
vocab_data['word2id']['<GO>'] = VOCAB_GO_ID
vocab_data['id2word'][VOCAB_PAD_ID] = '<PAD>'
vocab_data['id2word'][VOCAB_GO_ID] = '<GO>'


def _get_corpus_metrics():
    '''
    max length of questions
    '''
    for cat, data in zip(["valid", "test", "train"], [_valid_data, _test_data, _train_data]):
        max_len_question = 0
        total_len_question = 0
        max_len_utterance = 0
        total_len_utterance = 0
        for x in data:
            total_len_question += len(x['question'])
            total_len_utterance += len(x['utterance'])
            if len(x['question']) > max_len_question:
                max_len_question = len(x['question'])
            if len(x['utterance']) > max_len_utterance:
                max_len_utterance = len(x['utterance'])
        print('max len of %s question : %d, average: %d' % (cat, max_len_question, total_len_question/len(data)))
        print('max len of %s utterance: %d, average: %d' % (cat, max_len_utterance, total_len_utterance/len(data)))
    # max length of answers


class BatchIter():
    '''
    Load data with mini-batch
    '''
    def __init__(self, data = None, batch_size = 100):
        assert data is not None, "data should not be None."
        self.batch_size = batch_size
        self.data = data

    def next(self):
        random.shuffle(self.data)
        index = 0
        total_num = len(self.data)
        while index <= total_num:
            yield self.data[index:index + self.batch_size]
            index += self.batch_size

def padding(lis, pad, size):
    '''
    right adjust a list object
    '''
    if size > len(lis):
        lis += [pad] * (size - len(lis))
    else:
        lis = lis[0:size]
    return lis

def pack_question_n_utterance(q, u, q_length = 20, u_length = 99):
    '''
    combine question and utterance as input data for feed-forward network
    '''
    assert len(q) > 0 and len(u) > 0, "question and utterance must not be empty"
    q = padding(q, VOCAB_PAD_ID, q_length)
    u = padding(u, VOCAB_PAD_ID, u_length)
    assert len(q) == q_length, "question should be pad to q_length"
    assert len(u) == u_length, "utterance should be pad to u_length"
    return q + [VOCAB_GO_ID] + u

def __resolve_input_data(data, batch_size, question_max_length = 20, utterance_max_length = 99):
    '''
    resolve input data
    '''
    batch_iter = BatchIter(data = data, batch_size = batch_size)

    for mini_batch in batch_iter.next():
        result = []
        for o in mini_batch:
            x = pack_question_n_utterance(o['question'], o['utterance'], question_max_length, utterance_max_length)
            y_ = o['label']
            assert len(x) == utterance_max_length + question_max_length + 1, "Wrong length afer padding"
            assert VOCAB_GO_ID in x, "<GO> must be in input x"
            assert len(y_) == 2, "desired output."
            result.append([x, y_])
        if len(result) > 0:
            # print('data in batch:%d' % len(mini_batch))
            yield result
        else:
            raise StopIteration

# export data

def load_train(batch_size = 100, question_max_length = 20, utterance_max_length = 99):
    '''
    load train data
    '''
    result = []
    for o in _train_data:
        x = pack_question_n_utterance(o['question'], o['utterance'], question_max_length, utterance_max_length)
        y_ = o['label']
        assert len(x) == utterance_max_length + question_max_length + 1, "Wrong length afer padding"
        assert VOCAB_GO_ID in x, "<GO> must be in input x"
        assert len(y_) == 2, "desired output."
        result.append((x, y_))
    return result
    # return __resolve_input_data(_train_data, batch_size, question_max_length, utterance_max_length)

def load_test(question_max_length = 20, utterance_max_length = 99):
    '''
    load test data
    '''
    result = []
    for o in _test_data:
        x = pack_question_n_utterance(o['question'], o['utterance'], question_max_length, utterance_max_length)
        y_ = o['label']
        assert len(x) == utterance_max_length + question_max_length + 1, "Wrong length afer padding"
        assert VOCAB_GO_ID in x, "<GO> must be in input x"
        assert len(y_) == 2, "desired output."
        result.append((x, y_))
    return result

def load_valid(batch_size = 100, question_max_length = 20, utterance_max_length = 99):
    '''
    load valid data
    '''
    result = []
    for o in _valid_data:
        x = pack_question_n_utterance(o['question'], o['utterance'], question_max_length, utterance_max_length)
        y_ = o['label']
        assert len(x) == utterance_max_length + question_max_length + 1, "Wrong length afer padding"
        assert VOCAB_GO_ID in x, "<GO> must be in input x"
        assert len(y_) == 2, "desired output."
        result.append((x, y_))
    return result
    # return __resolve_input_data(_valid_data, batch_size, question_max_length, utterance_max_length)

def test_batch():
    '''
    retrieve data with mini batch
    '''
    for mini_batch in load_test():
        x, y_ = mini_batch
        print("length", len(x))
        assert len(y_) == 2, "data size should be 2"

    print("VOCAB_PAD_ID", VOCAB_PAD_ID)
    print("VOCAB_GO_ID", VOCAB_GO_ID)

def main():
    import lightnn
    from lightnn.models import Model
    from lightnn.layers import Dense, Input
    from lightnn.base import optimizers
    import numpy as np
    from sklearn.metrics import confusion_matrix

    batch_size = 50
    lr = 1e-4
    question_max_length = 20
    utterance_max_length = 99
    train_X, train_y = zip(*load_train())
    valid_X, valid_y = zip(*load_valid())
    test_X, test_y = zip(*load_test())

    input = Input(input_shape=question_max_length + utterance_max_length + 1)
    d1 = Dense(100, activator='sigmoid')(input)
    d2 = Dense(50, activator='sigmoid')(d1)
    out = Dense(2, activator='softmax')(d2)
    model = Model(input, out)
    optimizer = optimizers.SGD(lr=lr)
    model.compile('CCE', optimizer=optimizer)
    model.fit(train_X, train_y, verbose=2, batch_size=batch_size, epochs=10,
              validation_data=[valid_X, valid_y])
    test_pred = model.predict(test_X)
    print(confusion_matrix(np.argmax(test_y, axis=-1), np.argmax(test_pred, axis=-1)))
    print(np.mean(np.equal(np.argmax(test_pred, axis=-1), np.argmax(test_y, axis=-1))))

if __name__ == '__main__':
    # test_batch()
    main()