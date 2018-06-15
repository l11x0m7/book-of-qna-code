# -*- encoding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


cutoff = 1e-12
def _cutoff(z):
    return np.clip(z, cutoff, 1 - cutoff)


class Loss(object):
    @staticmethod
    def forward(y_hat, y):
        raise NotImplementedError('forward function must be implemented.')

    @staticmethod
    def backward(y_hat, y):
        raise NotImplementedError('forward function must be implemented.')


class MeanSquareLoss(Loss):
    """
    calculate the MSE
    """
    @staticmethod
    def forward(y_hat, y, norm=2):
        """
        the default vector norm is : sum(abs(x)**2) / 2
        :param y_hat: vector, output from your network
        :param y: vector, the ground truth
        :return: scalar, the cost
        """
        # return 0.5 * np.linalg.norm(y_hat - y, ord=norm)
        return 0.5 * np.sum(np.power(y_hat - y, 2))

    @staticmethod
    def backward(y_hat, y):
        return y_hat - y


class BinaryCategoryLoss(Loss):
    """
        二分类的log损失,主要用于前一层为sigmod层的情况
    """
    @staticmethod
    def forward(y_hat, y):
        y_hat = _cutoff(y_hat)
        y = _cutoff(y)
        return -np.mean(np.sum(np.nan_to_num(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)), axis=1))

    @staticmethod
    def backward(y_hat, y):
        """
        :param y_hat:
        :param y:
        :param activator:
        :return:
        """
        y_hat = _cutoff(y_hat)
        y = _cutoff(y)
        divisor = y_hat * (1 - y_hat)
        return (y_hat - y) / divisor


class LogLikelihoodLoss(Loss):
    """
        多分类的log loss, 主要用于前一层为softmax的情况
    """
    @staticmethod
    def forward(y_hat, y):
        assert (np.abs(np.sum(y_hat, axis=1) - 1.) < cutoff).all()
        assert (np.abs(np.sum(y, axis=1) - 1.) < cutoff).all()
        y_hat = _cutoff(y_hat)
        y = _cutoff(y)
        return -np.mean(np.sum(np.nan_to_num(y * np.log(y_hat)), axis=1))

    @staticmethod
    def backward(y_hat, y):
        """
        The loss partial by z is : y_hat * (y - y_hat) / (-1 / y_hat) = y_hat - y
        softmax + loglikelihoodCost == sigmoid + crossentropyCost
        :param y_hat:
        :param y:
        :param activator:
        :return:
        """
        assert (np.abs(np.sum(y_hat, axis=1) - 1.) < cutoff).all()
        assert (np.abs(np.sum(y, axis=1) - 1.) < cutoff).all()
        y_hat = _cutoff(y_hat)
        y = _cutoff(y)
        return y_hat - y


class FocalLoss(Loss):
    """
        Kaiming He提出的用于解决样本类别不均衡与其导致的easy sample dominating的问题
        一般前面接一个softmax层
    """
    # 这里我们使用默认的gama=2
    gama = 2

    @staticmethod
    def forward(y_hat, y):
        assert (np.abs(np.sum(y_hat, axis=1) - 1.) < cutoff).all()
        assert (np.abs(np.sum(y, axis=1) - 1.) < cutoff).all()
        y_hat = _cutoff(y_hat)
        y = _cutoff(y)
        return -np.mean(np.sum(np.nan_to_num(y * np.power((1 - y_hat), gama) * np.log(y_hat)), axis=1))

    @staticmethod
    def backward(y_hat, y):
        assert (np.abs(np.sum(y_hat, axis=1) - 1.) < cutoff).all()
        assert (np.abs(np.sum(y, axis=1) - 1.) < cutoff).all()
        y_hat = _cutoff(y_hat)
        y = _cutoff(y)
        return (-np.power((1 - y_hat), gama - 1) * y_hat * gama * np.log(y_hat) - np.power((1 - y_hat), gama)) * (y - y_hat)


# aliases from Keras
mse = MSE = MeanSquareLoss
cce = CCE = categorical_crossentropy = LogLikelihoodLoss
bce = BCE = binary_crossentropy = BinaryCategoryLoss
fce = FCE = FocalLoss


def get(loss):
    if isinstance(loss, str):
        loss = loss.lower()
        if loss in ('mse', 'meansquareloss'):
            return MSE()
        elif loss in ('categorical_crossentropy', 'cce', 'loglikelihoodloss'):
            return CCE()
        elif loss in ('binary_crossentropy', 'bce', 'binarycategoryloss'):
            return BCE()
        elif loss in ('fce', 'FCE', 'FocalLoss'):
            return FCE()
        else:
            raise ValueError('Unknown loss name `{}`'.format(loss))
    elif isinstance(loss, Loss):
        return loss
    else:
        raise ValueError('Unknown loss type `{}`'.format(loss.__class__.__name__))
