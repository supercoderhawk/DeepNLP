# -*- coding: UTF-8 -*-
import numpy as np
import re
import os
import collections
from utils import strQ2B
from config import CorpusType


class PrepareData:
  def __init__(self, vocab_size, corpus, dict_path=None, type=CorpusType.Train):
    self.vocab_size = vocab_size
    self.dict_path = dict_path
    self.SPLIT_CHAR = '  '  # 分隔符：双空格
    self.count = [['UNK', 0], ['STRT', 0],
                  ['END', 0]]  # 字符数量，其中'UNK'表示词汇表外的字符，'STAT'表示句子首字符之前的字符，'END'表示句子尾字符后面的字符，这两个字符用于生成字的上下文
    if type == CorpusType.Train:
      self.input_file = 'corpus/' + corpus + '_' + 'training.utf8'
      self.output_base = 'corpus/' + corpus + '_' + 'training_'
    elif type == CorpusType.Test:
      self.input_file = 'corpus/' + corpus + '_' + 'test.utf8'
      self.output_base = 'corpus/' + corpus + '_' + 'test_'
    self.sentences = self.read_sentences()

    if self.dict_path == None:
      self.dictionary = self.build_dictionary('corpus/' + corpus + '_dict.utf8')
    else:
      self.dictionary = self.read_dictionary()
    if self.dictionary == None:
      print('vocabulary size larger than dictionary size')
      exit(1)

    if type == CorpusType.Train:
      self.characters_index, self.labels_index = self.build_dataset()
      np.save(self.output_base + 'characters', self.characters_index)
      np.save(self.output_base + 'labels', self.labels_index)
    elif type == CorpusType.Test:
      self.raw_sentences = list(map(lambda s:s.replace(self.SPLIT_CHAR,''),self.sentences))
      if os.path.exists('corpus/'+corpus+'_test_labels.npy'):
        self.labels_index = np.load('corpus/'+corpus+'_test_labels.npy')
      else:
        _, self.labels_index = self.build_dataset()

  def read_sentences(self):
    file = open(self.input_file, 'r', encoding='utf-8')
    content = file.read()
    sentences = re.sub('[ ]+', self.SPLIT_CHAR, strQ2B(content)).splitlines()  # 将词分隔符统一为双空格
    sentences = list(filter(None, sentences))  # 去除空行
    file.close()
    return sentences

  def build_dictionary(self, output=None):
    dictionary = {}
    words = ''.join(self.sentences).replace(' ', '')
    vocab_count = len(collections.Counter(words))
    if vocab_count + 3 < self.vocab_size:
      return None
    self.count.extend(collections.Counter(words).most_common(self.vocab_size - 3))

    for word, _ in self.count:
      dictionary[word] = len(dictionary)
    if output != None:
      with open(output, 'w', encoding='utf8') as file:
        for ch, index in dictionary.items():
          file.write(ch + ' ' + str(index) + '\n')
    return dictionary

  def read_dictionary(self):
    dict_file = open(self.dict_path, 'r', encoding='utf-8')
    dict_content = dict_file.read().splitlines()
    dictionary = {}
    dict_arr = map(lambda item: item.split(' '), dict_content)
    for _, dict_item in enumerate(dict_arr):
      dictionary[dict_item[0]] = int(dict_item[1])
    dict_file.close()
    if len(dictionary) < self.vocab_size:
      return None
    else:
      reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
      for i in range(self.vocab_size, len(dictionary)):
        dictionary.pop(reverse_dictionary[i])
    return dictionary

  def build_dataset(self):
    sentence_index = []
    labels_index = []
    for sentence in self.sentences:
      sentence_label = []
      word_index = []
      words = sentence.strip().split(self.SPLIT_CHAR)
      for word in words:
        l = len(word)
        if l == 0:
          continue
        elif l == 1:
          sentence_label.append(0)
        else:
          sentence_label.append(1)
          sentence_label.extend([2] * (l - 2))
          sentence_label.append(3)
        for ch in word:
          index = self.dictionary.get(ch)
          if index != None:
            word_index.append(index)
          else:
            word_index.append(0)
      sentence_index.append(word_index)
      labels_index.append(sentence_label)
    return np.array(sentence_index), np.array(labels_index)


if __name__ == '__main__':
  PrepareData(4000, 'pku')
  PrepareData(4000, 'msr')
