# book-of-qna-code
code collections for the book of qna

## Chapter 4

### Dataset

WikiQA, TrecQA, InsuranceQA. For experiments, we use WikiQA.

#### data preprocess on WikiQA

`run preprocess_wiki.ipynb`

### Siamese-NN model

#### train model

`python siamese.py --train --model NN`

#### test model

`python siamese.py --test --model NN`

### Siamese-CNN model

#### train model

`python siamese.py --train --model CNN`

#### test model

`python siamese.py --test --model CNN`

### Siamese-RNN model

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

### word2vec

word2vec的简单实现。

`python word2vec.py`

### glove

glove的简单实现。

`python glove.py`

### fasttext

fasttext的简单实现。

`python fasttext.py`
