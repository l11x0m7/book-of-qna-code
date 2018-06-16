# CRF 示例程序
使用CRF模型实现命名实体识别。

如果使用本书附带的[docker镜像](https://hub.docker.com/r/chatopera/qna-book/)，所有依赖已经安装好，不需要再次安装。使用docker镜像运行程序的方式详见[文档](https://github.com/l11x0m7/book-of-qna-code/blob/master/README.md)。

## 依赖

* python3.6
* [sklearn-crfsuite==0.3.6](https://github.com/TeamHG-Memex/sklearn-crfsuite)
* nltk==3.3

## 执行训练和测试

```
python crf_sample.py
```


日志：

```
root@c0af3b5946b4:/app/ch2/crf# python crf_sample.py
             precision    recall  f1-score   support

      B-LOC      0.810     0.784     0.797      1084
      I-LOC      0.690     0.637     0.662       325
     B-MISC      0.731     0.569     0.640       339
     I-MISC      0.699     0.589     0.639       557
      B-ORG      0.807     0.832     0.820      1400
      I-ORG      0.852     0.786     0.818      1104
      B-PER      0.850     0.884     0.867       735
      I-PER      0.893     0.943     0.917       634

avg / total      0.809     0.787     0.796      6178

Fitting 3 folds for each of 50 candidates, totalling 150 fits
[Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  4.9min
[Parallel(n_jobs=-1)]: Done 150 out of 150 | elapsed: 16.7min finished
best params: {'c1': 0.20329301587456747, 'c2': 0.018103894656043132}
best CV score: 0.7496818270303766
model size: 1.12M
             precision    recall  f1-score   support

      B-LOC      0.797     0.779     0.788      1084
      I-LOC      0.681     0.643     0.661       325
     B-MISC      0.707     0.575     0.634       339
     I-MISC      0.680     0.600     0.637       557
      B-ORG      0.811     0.831     0.821      1400
      I-ORG      0.855     0.776     0.814      1104
      B-PER      0.851     0.884     0.867       735
      I-PER      0.899     0.943     0.921       634

avg / total      0.806     0.785     0.794      6178

Top likely transitions:
B-ORG  -> I-ORG   7.141253
I-ORG  -> I-ORG   6.893664
I-MISC -> I-MISC  6.763016
B-MISC -> I-MISC  6.639394
B-PER  -> I-PER   6.102409
B-LOC  -> I-LOC   5.365455
I-LOC  -> I-LOC   4.627761
...
```