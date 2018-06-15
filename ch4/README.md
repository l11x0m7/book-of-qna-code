# Chapter 4

该章节主要是介绍深度学习基础，并给出每个深度学习单元如何去构造一个QA网络。


## Requirements

```
pip install numpy
pip install tensorflow
pip install nltk
pip install jieba
```

### Dataset

实验中，我们使用WikiQA。

#### data preprocess on WikiQA

`python preprocess_wiki.py`

### Siamese-NN model

利用全连接层实现的一个pointwise的QA网络。

[To this repo](siamese_nn/)

### Siamese-CNN model

利用卷积层和池化层实现的一个pointwise的QA网络。

[To this repo](siamese_cnn/)

### Siamese-RNN model

利用LSTM或GRU实现的一个pointwise的QA网络。

[To this repo](siamese_rnn/)
