#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 <hailiang.hl.wang@gmail.com> All Rights Reserved
#
#
# Author: Hai Liang Wang
# Date: 2018-07-22:13:44:50
#
#===============================================================================

"""
    Hidden Markov Model Tokenizer
"""

__copyright__ = "Copyright (c) 2017 . All Rights Reserved"
__author__    = "Hai Liang Wang"
__date__      = "2017-07-22:13:44:50"


import os
import sys
import json
curdir = os.path.dirname(os.path.abspath(__file__))

import util as helper
from train import OUT_OF_OBS

class Tokenizer():
    '''
    HMM Tokenizer
    '''
    
    def __init__(self):
        self._hmm_meta = self.__load_metadata()
        self.states = self._hmm_meta['pi']
        self.observations = self._hmm_meta['observations']
        self.pi = self._hmm_meta['pi']
        self.A = self._hmm_meta['A']
        self.B = self._hmm_meta['B']
        
    def __load_metadata(self):
        '''
        Load parameters for HMM
        '''
        model = os.path.join(curdir, 'hmm.json')
        if not os.path.exists(model): raise BaseException("model %s does not exist." % model)
        with open(model, 'r') as f:
            return json.load(f)

    def __viterbi(self, obs):
        '''
        Viterbi algorithm for predict
        '''
        V = [{}]
        path = {}

        # Initialize base cases (t == 0)
        for y in self.states:
            V[0][y] = self.pi[y] * self.B[y][obs[0]]
            path[y] = [y]

        # Run Viterbi for t > 0
        for t in range(1,len(obs)):
            V.append({})
            newpath = {}

            for y in self.states:
                (prob, state) = max([(V[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]], y0) for y0 in self.states])
                V[t][y] = prob
                newpath[y] = path[state] + [y]

            helper.DEBUG("t%d's path: %s" % (t, newpath))
            # Don't need to remember the old paths
            path = newpath

        (prob, state) = max([(V[len(obs) - 1][y], y) for y in self.states])
        return (prob, path[state])

    def __decode_sbme(self, text, labels, punctuations = None):
        '''
        map SBME tokens to segmented text
        '''
        helper.DEBUG('__decode_sbme %s' % punctuations)

        def resolve_punctuation(i):
            if punctuations and i in punctuations:
                return punctuations[i]

        token = ''
        for index, label in enumerate(labels):
            if label == 'S':
                yield text[index]
            if label == 'B': token = text[index]
            if label == 'M': token += text[index]
            if label == 'E':
                token += text[index]
                yield ''.join(token)
                token = ''

            # label的最后不是 E，但token有值的情况
            if index == (len(labels) - 1) and (labels[-1] != 'E') and token:
                yield token

            # 标点符号
            ps = resolve_punctuation(index+1)
            if ps:
                for x in ps: yield x

    def cut(self, sentence, punctuation = True):
        '''
        分词
        '''
        sentence = ''.join(sentence.split()) # remove whitespaces
        text = []
        punctuations = {}
        for ch in sentence:
            if helper.is_zh(ch): 
                if ch in self.observations: text.append(ch)
                else: text.append(OUT_OF_OBS)
            elif helper.is_punct(ch):
                if not len(text) in punctuations: punctuations[len(text)] = [ch] 
                else: punctuations[len(text)] += ch

        if len(text) > 0:
            prob, path = self.__viterbi(text)
            helper.DEBUG("Final path: %s" % path)
            return self.__decode_sbme(text, path, punctuations if punctuation > 0 else None)

        '''
        condition there are only punctuations in sentence
        '''
        if len(text) == 0 and len(punctuations.keys()) > 0:
            result = []
            for x in punctuations.values():
                [ result.append(y) for y in x ]
            return result

        return []

def test():
    print('hmm tokenizer')
    T = Tokenizer()
    print(' '.join(T.cut('作为市长，我也体会到这种危险。')))

if __name__ == '__main__':
    test()
