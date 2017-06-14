# -*- coding: UTF-8 -*-
import numpy as np
from dnn import DNN
from prepare_data import PrepareData
from config import CorpusType


def estimate_cws(current_labels, correct_labels):
  cor_dict = {}
  curt_dict = {}
  curt_start = 0
  cor_start = 0
  for label_index, (curt_label, cor_label) in enumerate(zip(current_labels, correct_labels)):
    if cor_label == 0:
      cor_dict[label_index] = label_index + 1
    elif cor_label == 1:
      cor_start = label_index
    elif cor_label == 3:
      cor_dict[cor_start] = label_index + 1

    if curt_label == 0:
      curt_dict[label_index] = label_index + 1
    elif curt_label == 1:
      curt_start = label_index
    elif curt_label == 3:
      curt_dict[curt_start] = label_index + 1

  cor_count = 0
  recall_length = len(curt_dict)
  prec_length = len(cor_dict)
  for curt_start in curt_dict.keys():
    if curt_start in cor_dict and curt_dict[curt_start] == cor_dict[curt_start]:
      cor_count += 1

  return cor_count, prec_length, recall_length


def evaluate_dnn():
  cws = DNN('lstm', is_seg=True)
  model = 'tmp/lstm-bmodel2.ckpt'
  print(cws.seg('小明来自南京师范大学', model, debug=True))
  print(cws.seg('小明是上海理工大学的学生', model))
  print(cws.seg('迈向充满希望的新世纪', model))
  print(cws.seg('我爱北京天安门', model))
  pre = PrepareData(4000, 'pku', dict_path='corpus/pku_dict.utf8', type=CorpusType.Test)
  sentences = pre.raw_sentences
  labels = pre.labels_index
  corr_count = 0
  re_count = 0
  total_count = 0

  for _, (sentence, label) in enumerate(zip(sentences, labels)):
    _, tag = cws.seg(sentence, model)
    cor_count, prec_count, recall_count = estimate_cws(tag, np.array(label))
    corr_count += cor_count
    re_count += recall_count
    total_count += prec_count
  prec = corr_count / total_count
  recall = corr_count / re_count

  print(prec)
  print(recall)
  print(2 * prec * recall / (prec + recall))


if __name__ == '__main__':
  evaluate_dnn()
