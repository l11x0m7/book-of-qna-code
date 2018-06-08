
# Chapter 7

该章节主要介绍社区问答中的问答匹配问题，并介绍具有代表性的几个深度匹配模型。在该章中我们给出一个简单易用的pairwise的问答匹配网络QACNN。


## Requirements

```
pip install numpy
pip install tensorflow
pip install nltk
pip install stanfordcorenlp
pip install jieba
```

### Dataset

实验中，我们使用WikiQA。

#### data preprocess on WikiQA

`run preprocess_wiki.ipynb`

### QACNN

给定一个问题，一个正确答案和一个错误答案，这个模型能够给出两个答案的排序（谁的得分比谁高）。

#### train model

`python qacnn.py --train`

#### test model

`python qacnn.py --test`
