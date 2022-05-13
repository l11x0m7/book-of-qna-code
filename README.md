## 快速购书

<p align="center">
  <b>快速购书<a href="https://item.jd.com/12479014.html" target="_blank">链接</a></b><br>
  <a href="https://item.jd.com/12479014.html" target="_blank">
  <img src="https://user-images.githubusercontent.com/3538629/48657619-bcd24880-ea6e-11e8-8c4e-8bcb00761942.png" width="400">
  </a>
</p>

## 在线讲解

《智能问答与深度学习》的在线课程由本书作者团队，CSDN 学院和电子工业出版社联合发布【[详情链接](https://edu.csdn.net/bundled/detail/59?utm_source=tg16)】。

<p align="center">
  <b>从0开始深度学习<a href="https://edu.csdn.net/bundled/detail/59?utm_source=tg16" target="_blank">链接</a></b><br>
  <a href="https://edu.csdn.net/bundled/detail/59?utm_source=tg16" target="_blank">
  <img src="https://user-images.githubusercontent.com/3538629/63840877-58a4c380-c9b4-11e9-9a50-5601e911b5de.png" width="600">
  </a>
</p>

## 精彩书评

```
本书不是简单的罗列算法，而是指向了如何让计算机处理语言这样一个有挑战性的话题，以最终构造一个问答系统为目标，充满了好奇和实践精神，是陪伴读者学习人工智能和语言处理的好书。

-- 王小川，搜狗CEO
```

```
本书介绍了近年来自然语言处理、信息检索系统和机器阅读理解的成果，带有翔实的示例，对实际应用有很好的借鉴意义，而且从原理上进行了解释，可以帮助读者掌握这些技术，是入门自然语言处理和深度学习的好书。

-- 李纪为，香侬科技CEO
```

# 《智能问答与深度学习》随书附带源码

**《智能问答与深度学习》** 这本书是服务于准备入门机器学习和自然语言处理的学生和软件工程师的，在理论上介绍了很多原理、算法，同时也提供很多示例程序增加实践性，这些程序被汇总到示例程序代码库，这些程序主要是帮助大家理解原理和算法的，欢迎大家下载和执行。代码库的地址是：

<https://github.com/l11x0m7/book-of-qna-code>

在阅读本书的过程中，各章有示例程序的段落会说明对应代码库的路径。同时，在代码库中，也有文档介绍如何执行程序。

## 安装依赖软件

快速执行源码的最佳实践是通过 docker 容器，读者需要在计算机中安装

- Git

Git 是一个分布式版本管理工具，目前很多开源码项目使用它发布和协作，下载地址：

<https://git-scm.com/>

- Docker

Docker 是容器技术，容器是一种构建、发布和执行软件服务的标准，容器能屏蔽操作系统的不一致性，简便了软件发布、开发和运维，下载地址：

<https://www.docker.com/>

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

初次运行该脚本时，会下载 docker 的镜像，在这个镜像中，我们安装了示例代码执行需要的依赖环境，这一步骤可能占用半个小时或更长时间，程序执行完毕，命令行终端会自动进入容器内部，如下图：

<img src="./assets/ch1-1.png" width="500">

至此，读者就具有可执行示例程序的环境了，详细使用说明参考各项目文件夹内的文档。

## 引用本书

BibTex Source 格式书写：

- English

```
@Book{qnadlbook2019,
  author    = {Hailiang Wang, Zhuohuan Li, Xuming Lin, Kexin Chen, Sizhen Lee},
  editor    = {Liujie Zheng},
  publisher = {Beijing:Publishing House of Electronics Industry},
  title     = {Intelligent Question-Answer System and Deep Learning},
  year      = {2019},
  edition   = {1},
  isbn      = {9787121349218},
  url       = {https://github.com/l11x0m7/book-of-qna-code},
}
```

- 中文

```
@Book{qnadlbook2019,
  author    = {王海良, 李卓桓, 林旭鸣, 陈可心, 李思珍},
  editor    = {郑柳洁},
  publisher = {电子工业出版社},
  title     = {智能问答与深度学习},
  year      = {2019},
  edition   = {1},
  isbn      = {9787121349218},
  url       = {https://github.com/l11x0m7/book-of-qna-code},
}
```

## 联系我们

在您遇到关于软件安装、容器运行、程序代码执行等问题时，可通过下面地址反馈给我们：

<https://github.com/l11x0m7/book-of-qna-code/issues>

## 第二章 机器学习基础

[马尔可夫链](ch2/markov)

[隐马尔可夫模型](ch2/hmm)

[CRF 模型](ch2/crf)

## 第三章 自然语言处理基础

[有向无环图(DAG)](ch3/DAG)

[MMSEG 中文分词器](ch3/mmseg)

[HMM 中文分词器](ch3/hmmseg)

[依存关系分析之 transition-based 经典算法](ch3/dependency-parser-nivre)

[依存关系分析之 transition-based 神经网络算法](ch3/dependency-parser-neural)

[Apache Lucene 示例程序](ch3/lucene-sample)

[Elasticsearch 信息检索](ch3/search-engine)

## 第四章 深度学习初步

[lightnn：教学用神经网络工具包](ch4/lightnn/)

[Siamese 神经网络完成问答任务](ch4/siamese_nn/)

[Siamese 卷积神经网络完成问答任务](ch4/siamese_cnn/)

[Siamese 循环神经网络完成问答任务](ch4/siamese_rnn/)

## 第五章 词向量实现及应用

该章节主要为大家介绍深度学习在自然语言处理中必不可少的部分：embedding。此处我们为大家介绍了三种比较经典的词向量模型：word2vec，glove 以及 fasttext。通过实现这三个模型，并在小数据集上测试，帮助大家更好的理解这三个模型的原理。

[N 元模型(ngrams)](ch5/ngrams)

[word2vec 的简单实现](ch5/word2vec)

[glove 的简单实现](ch5/glove)

[fasttext 的简单实现](ch5/fasttext)

## 第六章 社区问答中的 QA 匹配

该章节主要介绍社区问答中的问答匹配问题，并介绍具有代表性的几个深度匹配模型。在该章中我们给出一个简单易用的 pairwise 的问答匹配网络 QACNN。

[Pairwise 形式的 QACNN 模型](ch6/QACNN/)

[Decomposable Attention 模型](ch6/decomposable_att_model/)：复现《A Decomposable Attention Model for Natural Language Inference》

[多比较方式的比较-集成模型](ch6/seq_match_seq/)：复现《A COMPARE-AGGREGATE MODEL FOR MATCHING TEXT SEQUENCES》

[BiMPM 模型](ch6/bimpm/)：复现《Bilateral Multi-Perspective Matching for Natural Language Sentence》

## License

[Apache 2.0](./LICENSE)

[![chatoper banner][co-banner-image]][co-url]

[co-banner-image]: https://user-images.githubusercontent.com/3538629/42217321-3d5e44f6-7ef7-11e8-94e7-1574bfa1dbb8.png
[co-url]: https://www.chatopera.com

## Chatopera 云服务

[https://bot.chatopera.com/](https://bot.chatopera.com/)

[Chatopera 云服务](https://bot.chatopera.com)是一站式实现聊天机器人的云服务，按接口调用次数计费。Chatopera 云服务是 [Chatopera 机器人平台](https://docs.chatopera.com/products/chatbot-platform/index.html)的软件即服务实例。在云计算基础上，Chatopera 云服务属于**聊天机器人即服务**的云服务。

Chatopera 机器人平台包括知识库、多轮对话、意图识别和语音识别等组件，标准化聊天机器人开发，支持企业 OA 智能问答、HR 智能问答、智能客服和网络营销等场景。企业 IT 部门、业务部门借助 Chatopera 云服务快速让聊天机器人上线！

<details>
<summary>展开查看 Chatopera 云服务的产品截图</summary>
<p>

<p align="center">
  <b>自定义词典</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530072-da92d600-d33e-11e9-8656-01c26caff4f9.png" width="800">
</p>

<p align="center">
  <b>自定义词条</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530091-e41c3e00-d33e-11e9-9704-c07a2a02b84e.png" width="800">
</p>

<p align="center">
  <b>创建意图</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530169-12018280-d33f-11e9-93b4-9db881cf4dd5.png" width="800">
</p>

<p align="center">
  <b>添加说法和槽位</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530187-20e83500-d33f-11e9-87ec-a0241e3dac4d.png" width="800">
</p>

<p align="center">
  <b>训练模型</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530235-33626e80-d33f-11e9-8d07-fa3ae417fd5d.png" width="800">
</p>

<p align="center">
  <b>测试对话</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530253-3d846d00-d33f-11e9-81ea-86e6d47020d8.png" width="800">
</p>

<p align="center">
  <b>机器人画像</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530312-6442a380-d33f-11e9-869c-85fb6a835a97.png" width="800">
</p>

<p align="center">
  <b>系统集成</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530281-4ecd7980-d33f-11e9-8def-c53251f30138.png" width="800">
</p>

<p align="center">
  <b>聊天历史</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530295-5856e180-d33f-11e9-94d4-db50481b2d8e.png" width="800">
</p>

</p>
</details>

<p align="center">
  <b>立即使用</b><br>
  <a href="https://bot.chatopera.com" target="_blank">
      <img src="https://static-public.chatopera.com/assets/images/64531083-3199aa80-d341-11e9-86cd-3a3ed860b14b.png" width="800">
  </a>
</p>
