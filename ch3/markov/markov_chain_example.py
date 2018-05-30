#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 <> All Rights Reserved
#
#
# File: ch3/markov/markov_chain_example.py
# Author: Hai Liang Wang
# Date: 2018-05-30:18:22:33
#
#===============================================================================


from __future__ import print_function
from __future__ import division

__copyright__ = "Copyright (c) 2017 . All Rights Reserved"
__author__    = "Hai Liang Wang"
__date__      = "2018-05-30:18:22:33"


import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
        # raise "Must be using Python 3"
else:
    unicode = str

import numpy as np
from matplotlib import pyplot

P = np.matrix([[1, 0, 0, 0, 0, 0],
               [1/4, 1/2, 0, 1/4, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [1/16, 1/4, 1/8, 1/4, 1/4, 1/16],
               [0, 0, 0, 1/4, 1/2, 1/4],
               [0, 0, 0, 0, 0, 1]])

v = np.matrix([[0, 0, 1, 0, 0, 0]])

# 数学运算
plot_data = []
for step in range(5):
    result = v * P**step
    plot_data.append(np.array(result).flatten())

# Convert the data format
plot_data = np.array(plot_data)

# 绘制图形
pyplot.figure(1)
pyplot.xlabel('Steps')
pyplot.ylabel('Probability')
lines = []
for i, shape in zip(range(6), ['x', 'h', 'H', 's', '8', 'r+']):
    line, = pyplot.plot(plot_data[:, i], shape, label="S%i" % (i+1))
    lines.append(line)
pyplot.legend(handles=lines, loc=1)
pyplot.savefig(os.path.join(curdir, "markov-predict.png"))