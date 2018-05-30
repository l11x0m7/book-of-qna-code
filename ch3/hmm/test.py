#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 <> All Rights Reserved
#
#
# File: ch3/markov/hmm/test.py
# Author: Hai Liang Wang
# Date: 2018-05-30:18:22:33
#
#===============================================================================

import unittest
import numpy as np
import hmm

def convert_observations_to_index(observations, label_index):
    '''
    observation to index
    '''
    lis = []
    for o in observations:
        lis.append(label_index[o])
    return lis


def generate_index_map(lables):
    '''
    index to state or observation
    '''
    index_label = {}
    label_index = {}
    i = 0
    for l in lables:
        index_label[i] = l
        label_index[l] = i
        i += 1
    return label_index, index_label

def convert_map_to_vector(map, label_index):
    v = np.empty(len(map), dtype=float)
    for e in map:
        v[label_index[e]] = map[e]
    return v


def convert_map_to_matrix(map, label_index1, label_index2):
    m = np.empty((len(label_index1), len(label_index2)), dtype=float)
    for line in map:
        for col in map[line]:
            m[label_index1[line]][label_index2[col]] = map[line][col]
    return m

class HMMTestCase(unittest.TestCase):

    def setUp(self):
        # 状态 
        self.states = ('健康', '感冒')
        # 观测状态
        self.observations = ('正常', '发冷', '发烧')
        self.start_probability = {'健康': 0.6, '感冒': 0.4}
        self.transition_probability = {
            '健康': {'健康': 0.7, '感冒': 0.3},
            '感冒': {'健康': 0.4, '感冒': 0.6},
        }

        self.emission_probability = {
            '健康': {'正常': 0.5, '发冷': 0.4, '发烧': 0.1},
            '感冒': {'正常': 0.1, '发冷': 0.3, '发烧': 0.6},
        }

        self.states_label_index, self.states_index_label = generate_index_map(self.states)
        self.observations_label_index, self.observations_index_label = generate_index_map(
            self.observations)


        print("states_label_index", self.states_label_index)
        print("states_index_label", self.states_index_label)
        print("observations_label_index", self.observations_label_index)
        print("observations_index_label", self.observations_index_label)

        self.A = convert_map_to_matrix(
            self.transition_probability,
            self.states_label_index,
            self.states_label_index)

        print("A", self.A)

        self.B = convert_map_to_matrix(
            self.emission_probability,
            self.states_label_index,
            self.observations_label_index)

        print("B", self.B)

        self.pi = convert_map_to_vector(self.start_probability, self.states_label_index)

        print("Pi", self.pi)

        self.hmm = hmm.HMM(self.A, self.B, self.pi)

    def tearDown(self):
        pass

    def testObservationProbForward(self):
        observations = ["正常", "发冷", "发烧"]
        observations_index = convert_observations_to_index(
            observations, self.observations_label_index)
        # observations_index = [0, 1, 1, 2, 2, 2, 2, 1, 0]
        print('observations_index', observations_index)

        prob = self.hmm.observation_prob_forward(observations_index)
        print("forward",prob)

    def testObservationProbBackward(self):
        observations = ["正常", "发冷", "发烧"]
        observations_index = convert_observations_to_index(
            observations, self.observations_label_index)
        print('observations_index', observations_index)

        prob = self.hmm.observation_prob_backward(observations_index)
        print("backward", prob)

    def testViterbi(self):
        observations = ["正常", "发冷", "发烧"]
        observations_index = convert_observations_to_index(
            observations, self.observations_label_index)
        V, p = self.hmm.viterbi(observations_index)

        print(" " *7,
            " ".join(
                ("%8s" %
                 self.observations_index_label[i]) for i in observations_index))

        for s in range(0, 2):
            print("%7s: " %
                self.states_index_label[s] +
                " ".join(
                    "%10s" %
                    ("%f" %
                     v) for v in V[s]))

    def testBaumWelchTrain(self):
        # run a baum_welch_train
        observations_data, states_data = self.hmm.simulate(100)
        print('observations_data', observations_data)
        print('states_data', states_data)
        guess = hmm.HMM(np.array([[0.5, 0.5],
                                [0.5, 0.5]]),
                        np.array([[0.3, 0.3, 0.3],
                                [0.3, 0.3, 0.3]]),
                        np.array([0.5, 0.5])
                        )
        guess.baum_welch_train(observations_data)
        states_out = guess.state_path(observations_data)[1]
        p = 0.0
        for s in states_data:
            if next(states_out) == s:
                p += 1

        print(p / len(states_data))

if __name__ == '__main__':
    unittest.main()