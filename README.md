# 智能问答与深度学习 开源码

**《智能问答与深度学习》** 这本书是服务于准备入门机器学习和自然语言处理的学生和软件工程师的，在理论上介绍了很多原理、算法，同时也提供很多示例程序增加实践性，这些程序被汇总到示例程序代码库，这些程序主要是帮助大家理解原理和算法的，欢迎大家下载和执行。代码库的地址是：

https://github.com/l11x0m7/book-of-qna-code

在阅读本书的过程中，各章有示例程序的段落会说明对应代码库的路径。同时，在代码库中，也有文档介绍如何执行程序。

## 安装依赖软件

快速执行源码的最佳实践是通过docker容器，读者需要在计算机中安装

* Git

Git是一个分布式版本管理工具，目前很多开源码项目使用它发布和协作，下载地址：

https://git-scm.com/

* Docker

Docker是容器技术，容器是一种构建、发布和执行软件服务的标准，容器能屏蔽操作系统的不一致性，简便了软件发布、开发和运维，下载地址：

https://www.docker.com/

这两个工具能兼容多种操作系统，我们强烈建议在阅读本书的第三章前，安装二者。

## 下载源码

在命令行终端，使用下面的方式下载源码：

```
git clone https://github.com/l11x0m7/book-of-qna-code.git book-of-qna-code
```

## 执行示例程序

启动容器：

```
cd book-of-qna-code
./admin/run.sh # Mac OSX, Linux, Unix
```

初次运行该脚本时，会下载docker的镜像，在这个镜像中，我们安装了示例代码执行需要的依赖环境，这一步骤可能占用半个小时或更长时间，程序执行完毕，命令行终端会自动进入容器内部，如下图：

<img src="./assets/ch1-1.png" width="500">

至此，读者就具有可执行示例程序的环境了，详细使用说明参考各项目文件夹内的文档。

## 联系我们

在您遇到关于软件安装、容器运行、程序代码执行等问题时，可通过下面地址反馈给我们：

https://github.com/l11x0m7/book-of-qna-code/issues


## 第二章 机器学习基础

[马尔可夫链](ch2/markov)

[隐马尔可夫模型](ch2/hmm)

[CRF模型](ch2/crf)

## 第三章 自然语言处理基础

[有向无环图(DAG)](ch3/DAG)

[MMSEG中文分词器](ch3/mmseg)

[HMM中文分词器](ch3/hmmseg)

[依存关系分析之transition-based经典算法](ch3/dependency-parser-nivre)

[依存关系分析之transition-based神经网络算法](ch3/dependency-parser-neural)

[Apache Lucene示例程序](ch3/lucene-sample)

[Elasticsearch信息检索](ch3/search-engine)

## 第四章 深度学习初步

[Siamese神经网络完成问答任务]()

[Siamese卷积神经网络完成问答任务]()

[Siamese循环神经网络完成问答任务]()

## 第五章 词向量实现及应用

[N元模型(ngrams)](ch5/ngrams)

[word2vec](ch5/word2vec)

[glove](ch5/glove)

[fasttext](ch5/fasttext)

## 第六章 社区问答中的QA匹配

[Pairwise形式的模型:QACNN]()

## License
[Apache 2.0](./LICENSE)
