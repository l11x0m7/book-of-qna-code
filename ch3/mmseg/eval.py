#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 <> All Rights Reserved
#
#
# Author: Hai Liang Wang
# Date: 2018-07-05:16:28:10
#
#===============================================================================

"""
   
"""
from __future__ import print_function
from __future__ import division

__copyright__ = "Copyright (c) 2017 . All Rights Reserved"
__author__    = "Hai Liang Wang"
__date__      = "2018-06-05:16:28:10"


import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    sys.setdefaultencoding("utf-8")
    reload(sys)
    # raise "Must be using Python 3"
else:
    unicode = str

from tqdm import trange
from mmseg import Tokenizer

MT = Tokenizer()

def evaluate(tokenizer, input, output):
    output_lines = []
    input_lines = []
    with open(input, 'r') as f:
        for x in f.readlines():
            input_lines.append(x)
    
    for x in trange(len(input_lines)):
        # print("seg: %s" % input_lines[x])
        o = []
        for y in tokenizer.cut(input_lines[x]):
            if y.strip(): o.append(y.strip())
        output_lines.append(' '.join(o) + '\n')
        
    print("done.")

    with open(output, 'w') as fr:
        fr.writelines(output_lines)

def main():
    evaluate(MT, '/tools/icwb2-data/testing/msr_test.utf8', 'mm.msr_test.seg')

if __name__ == '__main__':
    main()

