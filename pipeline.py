# -*- coding: UTF-8 -*-
import numpy as np
from dnn import DNN
from config import TrainMode
from re_cnn import RECNN
from evaluate import estimate_ner


def get_cws(content, model_name):
  dnn = DNN('mlp', mode=TrainMode.Sentence, task='ner')
  ner = dnn.seg(content, model_path=model_name, ner=True, seq=True)[0]
  return ner


def get_ner(content, model_name):
  if model_name.startswith('mlp'):
    dnn = DNN('mlp', mode=TrainMode.Sentence, task='ner')
  else:
    dnn = DNN('lstm', mode=TrainMode.Batch, task='ner')
  ner = dnn.seg(content, model_path=model_name, ner=True, seq=True)[0]
  return ner


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

  prec = corr_count / prec_count
  recall = corr_count / recall_count
  f1 = 2 * prec * recall / (prec + recall)
  print('precision:', prec)
  print('precision:', recall)
  print('F1 score:', f1)


def evaluate_re():
  re_two = RECNN(2)
  re_multi = RECNN(29)
  re_two.evaluate('cnn_emr_model3.ckpt')
  re_multi.evaluate('cnn_emr_model3.ckpt')


if __name__ == '__main__':
  evaluate_ner('mlp-ner-model50.ckpt')
  evaluate_ner('lstm-ner-model50.ckpt')
  evaluate_re()
