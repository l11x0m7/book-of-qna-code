#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot

P = np.matrix([[1, 0, 0, 0, 0, 0],
               [1/4, 1/2, 0, 1/4, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [1/16, 1/4, 1/8, 1/4, 1/4, 1/16],
               [0, 0, 0, 1/4, 1/2, 1/4],
               [0, 0, 0, 0, 0, 1]])

v = np.matrix([[0, 0, 1, 0, 0, 0]])

# Get the data
plot_data = []
for step in range(5):
    result = v * P**step
    plot_data.append(np.array(result).flatten())

# Convert the data format
plot_data = np.array(plot_data)

# Create the plot
pyplot.figure(1)
pyplot.xlabel('Steps')
pyplot.ylabel('Probability')
lines = []
for i, shape in zip(range(6), ['x', 'h', 'H', 's', '8', 'r+']):
    line, = pyplot.plot(plot_data[:, i], shape, label="S%i" % (i+1))
    lines.append(line)
pyplot.legend(handles=lines, loc=1)
pyplot.show()