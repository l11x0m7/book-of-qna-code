#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 <stakeholder> All Rights Reserved
#
#
# Author: Hai Liang Wang
# Date: 2018-07-22:16:27:53
#
#===============================================================================

"""
   TODO: Module comments at here
   
   
"""

__copyright__ = "Copyright (c) 2017 . All Rights Reserved"
__author__    = "Hai Liang Wang"
__date__      = "2017-07-22:16:27:53"


import os
import sys
import unittest
curdir = os.path.dirname(os.path.abspath(__file__))

from hmm import Tokenizer
import util as helper

class HMMSegTest(unittest.TestCase):

    def setUp(self):
        self.T = Tokenizer()

    def test_punctuation(self):
        print(' '.join(self.T.cut('作为市长，我也体会到这种危险。', punctuation = False)))

    def test_basecase(self):
        print(' '.join(self.T.cut('长春市长春节致词。', punctuation = True)))

    def test_seg(self):
        print(' '.join(self.T.cut('＊  ＊  ＊  ＊  ＊')))

    def test_oov(self):
        print(' '.join(self.T.cut('温济泽')))
        print(' '.join(self.T.cut('桑新')))

if __name__ == '__main__':
    unittest.main()
