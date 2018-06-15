# Chapter 6

该章节主要介绍社区问答中的问答匹配问题，并介绍具有代表性的几个深度匹配模型。在该章中我们给出一个简单易用的pairwise的问答匹配网络QACNN。

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

### QACNN

给定一个问题，一个正确答案和一个错误答案，这个模型能够给出两个答案的排序（谁的得分比谁高）。

[To this repo](qacnn/)

### Decomposable Attention Model

复现《A Decomposable Attention Model for Natural Language Inference》

[To this repo](decomposable_att_model/)

### Compare-Aggregate Model with Multi-Compare

复现《A COMPARE-AGGREGATE MODEL FOR MATCHING TEXT SEQUENCES》

[To this repo](seq_match_seq/)

### BiMPM

复现《Bilateral Multi-Perspective Matching for Natural Language Sentence》

[To this repo](bimpm/)

