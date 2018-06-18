# Siamese循环神经网络完成问答任务

利用LSTM或GRU实现的一个pointwise的QA网络。


## 依赖

* python2.7
* tensorflow==1.8.0

## 安装

如果使用本书附带的[docker镜像](https://hub.docker.com/r/chatopera/qna-book/)，所有依赖已经安装好，不需要再次安装。使用docker镜像运行程序的方式详见[文档](https://github.com/l11x0m7/book-of-qna-code/blob/master/README.md)。

### 下载词向量文件[glove](../download.sh)。

```
../download.sh
```

## 预处理wiki数据

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

某模型测试结果
```
root@feedd41cb9b4:/app/ch4/siamese_rnn# ./test.sh
test model
embedding file: /tools/embedding/glove.6B.100d.txt
Pre-trained: 27058 (92.94%)
[test] MAP:0.57135303545, MRR:0.587008845723
```

