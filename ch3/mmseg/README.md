# MMSEG分词算法

如果使用本书附带的[docker镜像](https://hub.docker.com/r/chatopera/qna-book/)，所有依赖已经安装好，不需要再次安装。

## UML

核心类的接口和类之间的关系

<img src="./classes.png" width="600">


## 测试分词器
```
python test.py
```

## 分词器评测
使用黄金标准进行评测

```
./evaluate.sh
```