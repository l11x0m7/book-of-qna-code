#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 <stakeholder> All Rights Reserved
#
#
# File: /Users/hain/ai/wordseg-algorithm/wordseg_dag.py
# Author: Hai Liang Wang
# Date: 2017-07-11:22:32:26
#
#===============================================================================

"""
   TODO: Module comments at here
   Python 3
"""

__copyright__ = "Copyright (c) 2017 . All Rights Reserved"
__author__    = "Hai Liang Wang"
__date__      = "2017-07-11:22:32:26"


import os
import sys
import re
from wordseg_dag import get_DAG
from math import log

re_eng = re.compile('[a-zA-Z0-9]', re.U)

# FIXME
# mocked finalseg
class Finalseg():
    def cut(self, buf):
        print("Finalseg: %s" % buf)
        return ["foo", "bar"]

finalseg = Finalseg()

dictionary = dict({
    "刘": [10, 'nz'],
    "刘德": [10, 'nz'],
    "刘德华": [20, 'nz'],
    "忘": [0, 'nz'],
    "忘情": [10, 'v'],
    "忘情水": [20, 'nz']
})

def calc(sentence, dag, route):
    N = len(sentence)
    route[N] = (0, 0)
    # 对概率值取对数之后的结果(可以让词频相除计算变成对数相减,防止除造成上溢)
    total = 0
    for x in dictionary.keys():
        total += dictionary[x][0]

    logtotal = log(total)
    # 从后往前遍历句子 反向计算最大概率
    for idx in range(N - 1, -1, -1):
        # 列表推倒求最大概率对数路径
        # route[idx] = max([ (概率对数，词语末字位置) for x in dag[idx] ])
        # 以idx:(概率对数最大值，词语末字位置)键值对形式保存在route中
        # route[x+1][0] 表示 词路径[x+1,N-1]的最大概率对数,
        # [x+1][0]即表示取句子x+1位置对应元组(概率对数，词语末字位置)的概率对数
        # route[x+1][0] 的数值都小于0，越接近0，代表概率越大
        resolve_weight = lambda x: dictionary[sentence[idx:x + 1]][0] \
            if sentence[idx:x + 1] in dictionary and \
            dictionary[sentence[idx:x + 1]][0] > 0 else 1
        route[idx] = max((log(resolve_weight(x)) - logtotal + route[x + 1][0], x) for x in dag[idx])

def cut_hmm(sentence):
    '''
    使用HMM模型分词
    '''
    dag = get_DAG(sentence, dictionary = dictionary)
    routes = {}
    calc(sentence, dag, routes)
    print(routes)

    x = 0
    N = len(sentence)
    buf = ''
    while x < N:
        y = routes[x][1] + 1
        l_word = sentence[x:y]
        # buf收集连续的单个字,把它们组合成字符串再交由 finalseg.cut函数来进行下一步分词
        if y - x == 1:
            buf += l_word
        else:
            if buf:
                if len(buf) == 1:
                    yield buf
                    buf = ''
                else:
                    if (not buf in dictionary) or dictionary[buf][0] == 0:# 未登录词,利用HMM
                        recognized = finalseg.cut(buf)
                        for t in recognized:
                            yield t
                    else:
                        for elem in buf:
                            yield elem
                    buf = ''
            yield l_word
        x = y

    if buf:
        if len(buf) == 1:
            yield buf
        elif (not buf in dictionary) or dictionary[buf][0] == 0: # 未登录词,利用HMM
            recognized = finalseg.cut(buf)
            for t in recognized:
                yield t
        else:
            for elem in buf:
                yield elem

sentence = "刘德华演唱了忘情水"
print(sentence)
print('/'.join(cut_hmm(sentence)))
