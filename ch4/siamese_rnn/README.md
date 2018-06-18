# Siamese循环神经网络完成问答任务

利用LSTM或GRU实现的一个pointwise的QA网络。

## 准备

#### 下载词向量文件[glove](../download.sh)。

```
cd ..
bash download.sh
```

#### 预处理wiki数据

```
cd ..
python preprocess_wiki.py
```

## 运行

```
bash run.sh
```
