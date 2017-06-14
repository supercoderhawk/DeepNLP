# -*- coding: UTF-8 -*-
import numpy as np
import re
import os
import collections
from utils import strQ2B
from config import CorpusType, TrainMode
import matplotlib.pyplot as plt


class PrepareData:
  def __init__(self, vocab_size, corpus, batch_length=40, batch_size=20, dict_path=None, mode=TrainMode.Batch,
               type=CorpusType.Train):
    self.vocab_size = vocab_size
    self.dict_path = dict_path
    self.batch_length = batch_length
    self.batch_size = batch_size
    self.SPLIT_CHAR = '  '  # 分隔符：双空格
    # 字符数量，
    # 其中'BATCH_PAD'表示构建batch时不足时补的字符，'UNK'表示词汇表外的字符，
    # 'STAT'表示句子首字符之前的字符，'END'表示句子尾字符后面的字符，这两个字符用于生成字的上下文
    self.count = [['BATCH_PAD', 0], ['UNK', 0], ['STRT', 0], ['END', 0]]
    self.init_count = len(self.count)
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
      if mode == TrainMode.Sentence:
        self.characters_index, self.labels_index = self.build_dataset()
        np.save(self.output_base + 'characters', self.characters_index)
        np.save(self.output_base + 'labels', self.labels_index)
      elif mode == TrainMode.Batch:
        self.character_batches, self.label_batches, self.lengths = self.build_batch()
        np.save(self.output_base + 'character_batches', self.character_batches)
        np.save(self.output_base + 'label_batches', self.label_batches)
        np.save(self.output_base + 'lengths', self.lengths)
    elif type == CorpusType.Test:
      self.raw_sentences = list(map(lambda s: s.replace(self.SPLIT_CHAR, ''), self.sentences))
      if os.path.exists('corpus/' + corpus + '_test_labels.npy'):
        self.labels_index = np.load('corpus/' + corpus + '_test_labels.npy')
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
    if vocab_count + self.init_count < self.vocab_size:
      return None
    self.count.extend(collections.Counter(words).most_common(self.vocab_size - self.init_count))

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

  def build_batch(self):
    character_batches = []
    label_batches = []
    lengths = []
    parts = []
    unknown = 4
    for line in self.sentences:
      sentences = line.split('。')
      sections = list(map(lambda s: s.strip().split(','), sentences))
      sections = list(filter(lambda s: len(s) > 0, list(map(lambda s: s.strip(), [j for i in sections for j in i]))))
      parts.extend(sections)

    sentence_count = len(parts)

    for part in parts:
      length = 0
      real_length = 0
      character_batch = []
      label_batch = []
      segments = part.split(self.SPLIT_CHAR)
      for segment in segments:
        segment_length = len(segment)
        if length + segment_length > self.batch_length:
          extra = [unknown] * (self.batch_length - length)
          character_batch.extend(extra)
          label_batch.extend(extra)
          real_length = length
          length = self.batch_length
          break
        elif length + segment_length <= self.batch_length:
          length += segment_length
          if segment_length == 0:
            continue
          elif segment_length == 1:
            label_batch.append(0)
          else:
            label_batch.append(1)
            label_batch.extend([2] * (segment_length - 2))
            label_batch.append(3)
          for ch in segment:
            index = self.dictionary.get(ch)
            if index != None:
              character_batch.append(index)
            else:
              character_batch.append(1)
          if length == self.batch_length:
            real_length = length
      if length < self.batch_length:
        extra = [unknown] * (self.batch_length - length)
        character_batch.extend(extra)
        label_batch.extend(extra)
        real_length = length

      lengths.append(real_length)
      character_batches.append(character_batch)
      label_batches.append(label_batch)

    extra_count = sentence_count % self.batch_size
    character_batches = np.array(character_batches[:-extra_count], dtype=np.int32).reshape(
      [-1, self.batch_size, self.batch_length])
    label_batches = np.array(label_batches[:-extra_count], dtype=np.int32).reshape(
      [-1, self.batch_size, self.batch_length])
    lengths = np.array(lengths[:-extra_count], dtype=np.int32).reshape([-1, self.batch_size])
    return character_batches, label_batches, lengths

  def plot_lengths(self, lengths):
    pre_i = lengths[0]
    count = []
    x = []
    j = 0
    for i in lengths:
      if pre_i == i:
        j += 1
      else:
        count.append(j)
        x.append(pre_i)
        j = 0
        pre_i = i

    print(len(list(filter(lambda l: l > self.batch_length, lengths))))
    print(len(lengths))
    x = range(len(count))
    plt.plot(x, count)
    plt.ylabel('长度')
    plt.show()


if __name__ == '__main__':
  pre = PrepareData(4000, 'pku', mode=TrainMode.Batch)
