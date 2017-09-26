# -*- coding: UTF-8 -*-
import numpy as np
from dnn import DNN
from config import TrainMode
from re_cnn import RECNN
from evaluate import estimate_ner


def get_cws(content, model_name):
  dnn = DNN('mlp', mode=TrainMode.Sentence, task='ner')
  ner = dnn.seg(content, model_path=model_name, ner=True, trans=True)[1]
  return ner


def get_ner(content, model_name):
  if model_name.startswith('tmp/mlp'):
    dnn = DNN('mlp', mode=TrainMode.Sentence, task='ner', is_seg=True)
  else:
    dnn = DNN('lstm', mode=TrainMode.Batch, task='ner', is_seg=True)
  ner = dnn.seg(content, model_path=model_name, ner=True, trans=True)
  return ner[1]


def get_relation():
  re = RECNN(2)
  re.evaluate('cnn_emr_model3.ckpt')
  re.evaluate('cnn_emr_model3.ckpt')


def evaluate_ner(model_name):
  base_folder = 'corpus/emr_ner_test_'
  labels = np.load(base_folder + 'labels.npy')
  characters = np.load(base_folder + 'characters.npy')
  corr_count = 0
  prec_count = 0
  recall_count = 0
  for ch, l in zip(characters, labels):
    c_count, p_count, r_count = estimate_ner(get_ner(ch, model_name), l)
    corr_count += c_count
    prec_count += p_count
    recall_count += r_count
  print(corr_count, prec_count, recall_count)
  prec = corr_count / prec_count
  recall = corr_count / recall_count
  f1 = 2 * prec * recall / (prec + recall)
  print('precision:', prec)
  print('recall:', recall)
  print('F1 score:', f1)


def evaluate_re():
  re_two = RECNN(2)
  # re_multi = RECNN(29)
  window_size = [ [3], [4], [2, 3], [3, 4], [2, 3, 4]]
  for w in window_size:
    print('window size:', w)
    name = 'cnn_emr_model100_{0}.ckpt'.format('_'.join(map(str, w)))
    re_two.evaluate(name)
    # re_multi.evaluate(name)


if __name__ == '__main__':
  # 实体识别
  # print('mlp')
  # evaluate_ner('tmp/mlp/mlp-ner-model50.ckpt')
  # print('mlp+embed')
  # evaluate_ner('tmp/mlp/mlp-ner-model50.ckpt')
  # print('lstm')
  # evaluate_ner('tmp/lstm/lstm-ner-model50.ckpt')
  # print('lstm+embed')
  # evaluate_ner('tmp/lstm/lstm-ner-model50.ckpt')
  # 关系抽取
  evaluate_re()
