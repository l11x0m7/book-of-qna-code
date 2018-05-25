
# Chapter 4

该章节主要是介绍深度学习基础，并给出每个深度学习单元如何去构造一个QA网络。

### Dataset

实验中，我们使用WikiQA。

#### data preprocess on WikiQA

`run preprocess_wiki.ipynb`

### Siamese-NN model

利用全连接层实现的一个pointwise的QA网络。

#### train model

`python siamese.py --train --model NN`

#### test model

`python siamese.py --test --model NN`

### Siamese-CNN model

利用卷积层和池化层实现的一个pointwise的QA网络。

#### train model

`python siamese.py --train --model CNN`

#### test model

`python siamese.py --test --model CNN`

### Siamese-RNN model

利用LSTM或GRU实现的一个pointwise的QA网络。

#### train model

`python siamese.py --train --model RNN`

#### test model

`python siamese.py --test --model RNN`

### QACNN

给定一个问题，一个正确答案和一个错误答案，这个模型能够给出两个答案的排序（谁的得分比谁高）。

注意：该模型是会在第七章中介绍。

#### train model

`python qacnn.py --train`

#### test model

`python qacnn.py --test`