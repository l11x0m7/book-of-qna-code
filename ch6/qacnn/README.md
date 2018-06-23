# QACNN

使用pairwise形式的QACNN网络实现问答任务，给定一个问题，一个正确答案和一个错误答案，这个模型能够给出两个答案的排序（谁的得分比谁高）。

## 安装

如果使用本书附带的[docker镜像](https://hub.docker.com/r/chatopera/qna-book/)，所有依赖已经安装好，不需要再次安装。使用docker镜像运行程序的方式详见[文档](https://github.com/l11x0m7/book-of-qna-code/blob/master/README.md)。

### python

* python2.7

使用pip安装python包
```
pip install -f ../requirements.txt
```

### 下载词向量文件[glove](../download.sh)。

```
../download.sh
```

## 预处理wiki数据

实验中，我们使用WikiQA数据集，数据集已经在 data 目录下，执行下面脚本进行预处理。

```
../preprocess.sh

```

正常运行后，有下面输出：

<img src="../assets/1.png" width="800">


## 运行

### 训练模型

```
./train.sh
```

### 测试模型

```
./test.sh
```

