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
from wordseg_dag import get_DAG


dictionary = dict({
    "刘": [10, 'nz'],
    "刘德": [10, 'nz'],
    "刘德华": [20, 'nz'],
    "忘": [0, 'nz'],
    "忘情": [10, 'v'],
    "忘情水": [20, 'nz']
})

def cut_all(sentence):
    '''
    输出DAG空间包含的所有词
    '''
    dag = get_DAG(sentence, dictionary = dictionary)
    old_j = -1
    for k, L in dag.items():
        if len(L) == 1 and k > old_j:
            yield sentence[k:L[0] + 1]
            old_j = L[0]
        else:
            for j in L:
                if j > k:
                    yield sentence[k:j + 1]
                    old_j = j    

sentence = "刘德华演唱了忘情水"
print('/'.join(cut_all(sentence)))