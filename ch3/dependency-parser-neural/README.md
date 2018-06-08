# 依存关系分析
使用神经网络训练依存关系分析模型

[项目继承自bist-parser](./bist-parser.README.md)

如果使用本书附带的[docker镜像](https://hub.docker.com/r/chatopera/qna-book/)，所有依赖已经安装好，不需要再次安装。

## 训练
Transition-based Parsing
```
./admin/train.sh
```

## 依赖
```
python2.7
pip
```

### 安装Python包
```
pip install -f ./requirements.txt
```



## 语料
格式 CoNLL-u format，进一步参考[text-dependency-parser](https://github.com/Samurais/text-dependency-parser).
