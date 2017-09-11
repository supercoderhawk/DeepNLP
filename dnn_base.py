# -*- coding: UTF-8 -*-
import numpy as np
from base import Base
from collections import OrderedDict


class DNNBase(Base):
  def __init__(self):
    Base.__init__(self)
    self.tags_count = 4
    self.dictionary = None
    self.skip_window_left = 1
    self.skip_window_right = 1
    self.window_size = self.skip_window_left + self.skip_window_right + 1
    self.hinge_discount = 0.2
    self.reverse_categories, self.category_reverse_dict, self.zh_categories = self.init_categories()

  def init_categories(self):
    categories = {'Sign': 'SN', 'Symptom': 'SYM', 'Part': 'PT', 'Property': 'PTY', 'Degree': 'DEG',
                  'Quality': 'QLY', 'Quantity': 'QNY', 'Unit': 'UNT', 'Time': 'T', 'Date': 'DT', 'Result': 'RES',
                  'Disease': 'DIS', 'DiseaseType': 'DIT', 'Examination': 'EXN', 'Location': 'LOC',
                  'Medicine': 'MED', 'Spec': 'SPEC', 'Usage': 'USG', 'Dose': 'DSE', 'Treatment': 'TRT',
                  'Family': 'FAM', 'Modifier': 'MOF'}
    zh_categories = {'Sign': '体征', 'Symptom': '症状', 'Part': '部位', 'Property': '性质', 'Degree': '程度',
                     'Quality': '定性值', 'Quantity': '定量值', 'Unit': '单位', 'Time': '时间', 'Date': '日期', 'Result': '结果',
                     'Disease': '疾病', 'DiseaseType': '疾病分型分歧', 'Examination': '检查', 'Location': '机构',
                     'Medicine': '药物', 'Spec': '规格', 'Usage': '用法', 'Dose': '用量', 'Treatment': '治疗',
                     'Family': '家族成员', 'Modifier': '其他修饰词'}
    category_labels_dict = OrderedDict({'O': 0})
    category_index = 1
    for category in categories:
      category_labels_dict[categories[category] + '_B'] = category_index
      category_index += 1
      category_labels_dict[categories[category] + '_O'] = category_index
      category_index += 1
    category_labels_dict['P'] = category_index
    return OrderedDict(zip(categories.values(), categories.keys())), OrderedDict(
      zip(category_labels_dict.values(), category_labels_dict.keys())), zh_categories

  def viterbi(self, emission, A, init_A, return_score=False, is_constraint=False, labels=None, size=4):
    """
    维特比算法的实现，所有输入和返回参数均为numpy数组对象
    :param emission: 发射概率矩阵，对应于本模型中的分数矩阵，4*length
    :param A: 转移概率矩阵，4*4
    :param init_A: 初始转移概率矩阵，4
    :param return_score: 是否返回最优路径的分值，默认为False
    :return: 最优路径，若return_score为True，返回最优路径及其对应分值
    """

    constraint = [[0, 1], [2, 3], [2, 3], [0, 1]]
    length = emission.shape[1]
    path = np.ones([self.tags_count, length], dtype=np.int32) * -1
    corr_path = np.zeros([length], dtype=np.int32)
    path_score = np.ones([self.tags_count, length], dtype=np.float64) * (np.finfo('f').min / 2)
    path_score[:, 0] = init_A + emission[:, 0]

    if labels is not None:
      for i in range(size):
        if i != labels[0]:
          path_score[i, 0] += self.hinge_discount

    for pos in range(1, length):
      for t in range(self.tags_count):
        for prev in range(self.tags_count):
          if is_constraint:
            if t not in constraint[prev]:
              continue
          temp = path_score[prev][pos - 1] + A[prev][t] + emission[t][pos]
          if labels is not None:
            if t != labels[pos]:
              temp += self.hinge_discount
          if temp >= path_score[t][pos]:
            path[t][pos] = prev
            path_score[t][pos] = temp

    max_index = np.argmax(path_score[:, -1])
    corr_path[length - 1] = max_index
    for i in range(length - 1, 0, -1):
      max_index = path[max_index][i]
      corr_path[i - 1] = max_index
    if return_score:
      return corr_path, path_score[max_index, -1]
    else:
      return corr_path

  def viterbi_new(self, emission, transition, transition_init, labels=None):
    constraint = [[0, 1], [2, 3], [2, 3], [0, 1]]
    length = emission.shape[1]
    path = np.ones([self.tags_count, length + 1], dtype=np.int32) * -1
    corr_path = np.zeros([length], dtype=np.int32)
    path_score = np.ones([self.tags_count, length + 1], dtype=np.float64) * (np.finfo('f').min / 2)
    # path_score[:, 0] = transition_init + emission[:, 0]
    path_score[0, 0] = 0

    for pos in range(1, length + 1):
      for path_index in range(self.tags_count):
        for curr_label in constraint[path_index]:
          tmp = path_score[path_index, pos - 1] + emission[curr_label, pos - 1] + transition[path_index, curr_label]
          if labels is not None:
            if curr_label != labels[pos - 1]:
              tmp += self.hinge_discount
          if tmp > path_score[curr_label, pos]:
            path_score[curr_label, pos] = tmp
            path[curr_label, pos] = path_index

    # print(path)
    # print(path_score)
    max_index = np.argmax(path_score[:, -1])
    corr_path[length - 1] = max_index
    for i in range(length - 1, 0, -1):
      max_index = path[max_index][i + 1]
      corr_path[i - 1] = max_index
    return corr_path

  def generate_transition_update(self, correct_tags, current_tags):
    if correct_tags.shape != current_tags.shape:
      print('序列长度不同')
      return None

    A_update = np.zeros([self.tags_count, self.tags_count], dtype=np.float32)
    init_A_update = np.zeros([self.tags_count], dtype=np.float32)
    before_corr = correct_tags[0]
    before_curr = current_tags[0]
    update_init = False

    if before_corr != before_curr:
      init_A_update[before_corr] += 1
      init_A_update[before_curr] -= 1
      update_init = True

    for _, (corr_tag, curr_tag) in enumerate(zip(correct_tags[1:], current_tags[1:])):
      if corr_tag != curr_tag or before_corr != before_curr:
        A_update[before_corr, corr_tag] += 1
        A_update[before_curr, curr_tag] -= 1
      before_corr = corr_tag
      before_curr = curr_tag

    return A_update, init_A_update, update_init

  def generate_transition_update_index(self, correct_labels, current_labels):
    if correct_labels.shape != current_labels.shape:
      print('序列长度不同')
      return None

    before_corr = correct_labels[0]
    before_curr = current_labels[0]
    update_init = False

    trans_init_pos = None
    trans_init_neg = None
    trans_pos = []
    trans_neg = []

    if before_corr != before_curr:
      trans_init_pos = [before_corr]
      trans_init_neg = [before_curr]
      update_init = True

    for _, (corr_label, curr_label) in enumerate(zip(correct_labels[1:], current_labels[1:])):
      if corr_label != curr_label or before_corr != before_curr:
        trans_pos.append([before_corr, corr_label])
        trans_neg.append([before_curr, curr_label])
      before_corr = corr_label
      before_curr = curr_label

    return trans_pos, trans_neg, trans_init_pos, trans_init_neg, update_init

  def sentence2index(self, sentence):
    index = []
    for word in sentence:
      if word not in self.dictionary:
        index.append(1)
      else:
        index.append(self.dictionary[word])

    return index

  def index2seq(self, indices):
    ext_indices = [2] * self.skip_window_left
    ext_indices.extend(indices + [3] * self.skip_window_right)
    seq = []
    for index in range(self.skip_window_left, len(ext_indices) - self.skip_window_right):
      seq.append(ext_indices[index - self.skip_window_left: index + self.skip_window_right + 1])

    return seq

  def tags2words(self, sentence, tags):
    words = []
    word = ''
    for tag_index, tag in enumerate(tags):
      if tag == 0:
        words.append(sentence[tag_index])
      elif tag == 1:
        word = sentence[tag_index]
      elif tag == 2:
        word += sentence[tag_index]
      else:
        words.append(word + sentence[tag_index])
        word = ''
    # 处理最后一个标记为I的情况
    if word != '':
      words.append(word)

    return words

  def tags2entities(self, sentence, tags):
    entities = []
    entity = ''
    for tag_index, tag in enumerate(tags):
      if tag == 0:
        continue
      elif tag == 1:
        entities.append(entity)
        entity = sentence[tag_index]
      else:
        entity += sentence[tag_index]
    if entity != '':
      entities.append(entity)
    return entities

  def tags2category_entities(self, sentence, tags):
    entities = []
    entity = ''
    category = ''
    for tag_index, tag in enumerate(tags):
      type = self.category_reverse_dict[tag]
      if tag == 0:
        continue
      elif type[-1] == 'B':
        entities.append(entity + '/' + category)
        entity = sentence[tag_index]
        category = self.zh_categories[self.reverse_categories[type[:-2]]]
      else:
        entity += sentence[tag_index]
    if entity != '':
      entities.append(entity + '/' + category)
    return entities
