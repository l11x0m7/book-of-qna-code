# 人工神经网络实现问答任务

基于[Tensorflow](https://www.tensorflow.org/)和[WikiQA数据集](https://aclweb.org/anthology/D15-1237)。


## 安装

如果使用本书附带的[docker镜像](https://hub.docker.com/r/chatopera/qna-book/)，所有依赖已经安装好，不需要再次安装。使用docker镜像运行程序的方式详见[文档](https://github.com/l11x0m7/book-of-qna-code/blob/master/README.md)。

#### 下载词向量文件[glove](../download.sh)。

```
../download.sh
```

## 预处理wiki数据

```
../preprocess.sh
```

正常运行后，有下面输出：

<img src="./assets/1.png" width="800">

## 运行

### 训练模型

```
./train.sh
```

### 测试模型

```
./test.sh
```
