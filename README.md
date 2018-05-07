# book-of-qna-code

code collections for the book of qna

## Chapter 4

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


## Chapter 5

该章节主要为大家介绍深度学习在自然语言处理中必不可少的部分：embedding。此处我们为大家介绍了三种比较经典的词向量模型：word2vec，glove以及fasttext。通过实现这三个模型，并在小数据集上测试，帮助大家更好的理解这三个模型的原理。

### word2vec

word2vec的简单实现。

`python word2vec.py`

### glove

glove的简单实现。

`python glove.py`

### fasttext

fasttext的简单实现。

`python fasttext.py`
