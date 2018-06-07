#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 <> All Rights Reserved
# 第二届自然语言处理与中文计算会议（NLP&CC 2013） 清华大学提供的数据格式和CoNLL-u 格式不一致。
# 本程序将其转化为 CoNLL-u 格式。
# File: /Users/hain/ai/text-dependency-parser/app/pio/thu_2013_data_format_converter.py
# Author: Hai Liang Wang
# Date: 2018-03-19:15:04:09
#
#===============================================================================

"""
   
"""
from __future__ import print_function
from __future__ import division

__copyright__ = "Copyright (c) 2017 . All Rights Reserved"
__author__    = "Hai Liang Wang"
__date__      = "2018-03-19:15:04:09"


import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"

# Get ENV
ENVIRON = os.environ.copy()



def conv(from_, to_):
    result = []
    with open(from_, "r") as fin:
        for x,y in enumerate(list(fin.readlines())):
            s = y.strip()
            if s:
                o = s.split("\t")
                assert len(o) == 8, "wrong text length"
                result.append("\t".join(o) + "\t_\t_\n")
            else:
                # print("index: %s | black" % x)
                result.append(y)
            # o = x.split("\t")
            # print("conv: %s" % x.strip())

    with open(to_, "w") as fout:
        fout.writelines(result)
        print("done %s" % to_)


import unittest

# run testcase: python /Users/hain/ai/text-dependency-parser/app/pio/thu_2013_data_format_converter.py Test.testExample
class Test(unittest.TestCase):
    '''
    
    '''
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_convert(self):
        print("test_convert")
        form_ = ["dev.conll", "train.conll"]
        to_ = os.path.join(curdir, os.path.pardir, os.path.pardir, "data", "evsam05", "THU")
        for x in form_:
            f = os.path.join(curdir, os.path.pardir, os.path.pardir, "data", "evsam05", "THU", x)
            t = "%su" % f
            conv(f, t)

def test():
    unittest.main()

if __name__ == '__main__':
    test()
