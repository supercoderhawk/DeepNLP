# 深度学习工具库

本项目基于`tensorflow`，实现一些论文提出的基于深度学习的分词、命名实体识别和实体关系抽取模型。

本项目是在[DNN_CWS](https://github.com/supercoderhawk/DNN_CWS)的基础上进行开发。增加了实体关系抽取。

**本项目目前有重构的计划**

**本项目迁移至[DeepLearning_NLP](https://github.com/supercoderhawk/DeepLearning_NLP)，故本项目暂时停止维护**

## 项目功能

* 中文分词
* 命名实体识别
* 实体关系抽取

## 依赖
1. python >= 3.5
2. tensorflow>=1.2.0
3. matplotlib>=1.5.3

## 语料库

文件夹`corpus`下：

1. pku_training.utf8、pku_test.utf8: sighan 2005 bakeoff 北大分词库
2. msr_training.utf8、msr_test.utf8: sighan 2005 bakeoff 微软亚洲研究院分词库
3. msr_ner_training.utf8: sighan 2006 bakeoff 微软亚洲研究院命名实体识别语料库
4. semeval_relation.utf8: International Workshop on Semantic Evaluation (SemEval)
 2010 task 8 关系抽取数据集

## 参考论文

### 中文分词 && 命名实体识别
* [deep learning for chinese word segmentation and pos tagging](www.aclweb.org/anthology/D13-1061) （已完全实现，文件[`dnn.py`](https://github.com/supercoderhawk/DeepNLP/blob/master/dnn.py)）
* [Long Short-Term Memory Neural Networks for Chinese Word Segmentation](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP141.pdf) (完全实现，需要调参，文件[`dnn.py`](https://github.com/supercoderhawk/DeepNLP/blob/master/dnn.py))
* [Max-Margin Tensor Neural Network for Chinese Word Segmentation](www.aclweb.org/anthology/P14-1028) （正在实现，文件[`mmtnn.py`](https://github.com/supercoderhawk/DeepNLP/blob/master/mmtnn.py)）

## 实体关系抽取
* [relation extraction: perspective from convolutional neural networks](http://aclweb.org/anthology/W15-1506) （已完全实现，文件[`re_cnn`](https://github.com/supercoderhawk/DeepNLP/blob/master/re_cnn.py)）
## TodoList

- [ ] 支持`pip`

