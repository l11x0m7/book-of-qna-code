# 隐马尔可夫模型

一阶隐马尔可夫模型，以及维特比算法、前后向算法和Baum-Welch算法的实现。

## 前向算法

* 预测：计算给定观测序列的发生概率

```
python test.py HMMTestCase.testObservationProbForward
```

## 后向算法

* 预测：计算给定观测序列的发生概率

```
python test.py HMMTestCase.testObservationProbBackward
```

前向算法与后向算法对同一个观测序列的预测概率是一致的，二者只是计算起点不同。二者的时间复杂度也是一致的。

## 维特比算法

* 解码：对于给定的观测序列，寻找最优的状态序列

```
python test.py HMMTestCase.testViterbi
```

## Baum Welch算法

* 模型参数估计：在给定可观测序列的基础上，使用EM算法学习HMM的参数

```
python test.py HMMTestCase.testBaumWelchTrain
```