![Version](https://img.shields.io/badge/Version-0.0.7-blue.svg) ![Version](https://img.shields.io/badge/Python-2.7-green.svg) ![Version](https://img.shields.io/badge/Numpy-1.13.0-yellow.svg) ![Version](https://img.shields.io/badge/Linux-x.x.x-red.svg)

# lightnn
The light(\`light\` means not many codes here) deep learning framework for study and for fun. Join us!

## How to install

### pip install

`pip install lightnn`

### python install

`python setup.py install`


## Modual structure

### models

* Sequential
* Model

### activations

* identity(None)
* sigmoid
* relu
* softmax
* tanh
* leaky relu
* elu
* selu
* thresholded relu
* softplus
* softsign
* hard sigmoid

### losses

* MeanSquareLoss
* BinaryCategoryLoss
* LogLikelihoodLoss
* FocalLoss

### initializers

* zeros
* ones
* xavier uniform initializer(glorot uniform initializer)
* default weight initializer
* large weight initializer
* orthogonal initializer

### optimizers

* SGD
* Momentum(Nestrov included)
* RMSProp
* Adam
* Adagrad
* Adadelta

### layers

* FullyConnected(Dense)
* Conv2d
* MaxPooling
* AvgPooling
* Softmax
* Dropout
* Flatten
* Activation
* RNN
* LSTM
* GRU

### utils

* label smoothing
* sparse to dense

### gradient check

* Dense
* CNN and Pooling
* RNN, LSTM and GRU


### examples

* MLP MNIST Classification
* CNN MNIST Classification
* RNN Language Model
* LSTM Language Model
* GRU Language Model

## Document instructions

* English for classes and functions
* Chinese for annotation


## References
1. [Keras](https://github.com/fchollet/keras): a polular deep learning framework based on tensorflow and theano.
2. [NumpyDL](https://github.com/oujago/NumpyDL): a simple deep learning framework with manual-grad, totally written with python and numpy.([Warning] Some errors in `backward` part of this project)
3. [paradox](https://github.com/ictxiangxin/paradox): a simple deep learning framework with symbol calculation system. Lightweight for learning and for fun. It's totally written with python and numpy.
4. [Bingtao Han's blogs](https://zybuluo.com/hanbingtao/): easy way to go for deep learning([Warning] Some calculation errors in `RNN` part).

