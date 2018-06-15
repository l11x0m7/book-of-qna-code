# -*-encoding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _check_convolution_layer(filter_size, filter_num, zero_padding, stride):
    # if not isinstance(input_height, int):
    #     raise ValueError('`input_height` must be int')
    # if not isinstance(input_width, int):
    #     raise ValueError('`input_width` must be int')
    # if not isinstance(input_channel, int):
    #     raise ValueError('`input_channel` must be int')
    filter_height, filter_width = filter_size
    if not isinstance(filter_height, int):
        raise ValueError('`filter_height` must be int')
    if not isinstance(filter_width, int):
        raise ValueError('`filter_width` must be int')
    if not isinstance(filter_num, int):
        raise ValueError('`filter_num` must be int')
    if not isinstance(zero_padding, tuple) and not isinstance(zero_padding, list):
        raise ValueError('`zero_padding` must be tuple(list) or int')
    # if input_height + zero_padding[0] * 2 < filter_height:
    #     raise ValueError('`filter_height` is too large')
    # if input_width + zero_padding[1] * 2 < filter_width:
    #     raise ValueError('`filter_width` is too large')
    # if stride[0] > filter_height:
    #     raise ValueError('`stride` is too large on axis 0')
    # if stride[1] > filter_width:
    #     raise ValueError('`stride` is too large on axis 1')
