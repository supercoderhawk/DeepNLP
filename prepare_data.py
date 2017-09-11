# -*- coding: UTF-8 -*-
import numpy as np
import re
import os
import collections
from utils import plot_lengths
from config import CorpusType, TrainMode


class PrepareData:
  def __init__(self, vocab_size, corpus, batch_length=100, batch_size=50, dict_path=None, mode=TrainMode.Batch,
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
    self.lines = self.read_lines()

    if self.dict_path == None:
      self.dictionary, self.reverse_dictionary = self.build_dictionary('corpus/' + corpus + '_dict.utf8')
    else:
      self.dictionary, self.reverse_dictionary = self.read_dictionary()
    if self.dictionary == None:
      print('vocabulary size larger than dictionary size')
      exit(1)

    if type == CorpusType.Train:
      self.characters_index, self.labels_index = self.build_dataset()
      if mode == TrainMode.Sentence:
        np.save(self.output_base + 'characters', self.characters_index)
        np.save(self.output_base + 'labels', self.labels_index)
      elif mode == TrainMode.Batch:
        self.character_batches, self.label_batches, self.lengths, self.sentences, self.sentence_labels, self.sentence_lengths = self.build_batch()
        np.save(self.output_base + 'character_batches', self.character_batches)
        np.save(self.output_base + 'label_batches', self.label_batches)
        np.save(self.output_base + 'lengths', self.lengths)
    elif type == CorpusType.Test:
      self.raw_lines = list(map(lambda s: s.replace(self.SPLIT_CHAR, ''), self.lines))
      if os.path.exists('corpus/' + corpus + '_test_labels.npy'):
        self.labels_index = np.load('corpus/' + corpus + '_test_labels.npy')
      else:
        _, self.labels_index = self.build_dataset()
        np.save('corpus/' + corpus + '_test_labels', self.labels_index)

    # plot_lengths(self.sentence_lengths)

  def read_lines(self):
    file = open(self.input_file, 'r', encoding='utf-8')
    content = file.read()
    # sentences = re.sub('[ ]+', self.SPLIT_CHAR, strQ2B(content)).splitlines()  # 将词分隔符统一为双空格
    sentences = re.sub('[ ]+', self.SPLIT_CHAR, content).splitlines()  # 将词分隔符统一为双空格
    sentences = list(map(lambda s: s.strip(), filter(None, sentences)))  # 去除空行，去首尾空格
    file.close()
    return sentences

  def build_dictionary(self, output=None):
    dictionary = {}
    words = ''.join(self.lines).replace(' ', '')
    vocab_count = len(collections.Counter(words))
    print('characters count'+str(vocab_count))
    if vocab_count + self.init_count < self.vocab_size:
      return None
    self.count.extend(collections.Counter(words).most_common(self.vocab_size - self.init_count))

    for word, _ in self.count:
      dictionary[word] = len(dictionary)
    if output != None:
      with open(output, 'w', encoding='utf8') as file:
        for ch, index in dictionary.items():
          file.write(ch + ' ' + str(index) + '\n')
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

  def read_dictionary(self):
    dict_file = open(self.dict_path, 'r', encoding='utf-8')
    dict_content = dict_file.read().splitlines()
    dictionary = {}
    dict_arr = map(lambda item: item.split(' '), dict_content)
    for _, dict_item in enumerate(dict_arr):
      dictionary[dict_item[0]] = int(dict_item[1])
    dict_file.close()
    # if len(dictionary) < self.vocab_size:
    #  return None
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # else:
    #   reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    #   for i in range(self.vocab_size, len(dictionary)):
    #     dictionary.pop(reverse_dictionary[i])
    return dictionary, reverse_dictionary

  def build_dataset(self):
    sentence_index = []
    labels_index = []
    for sentence in self.lines:
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
          if index is not None:
            word_index.append(index)
          else:
            word_index.append(0)
      sentence_index.append(word_index)
      labels_index.append(sentence_label)
    return np.array(sentence_index), np.array(labels_index)

  def build_batch(self):
    sentence_batches = []
    label_batches = []
    sentence_lengths = []
    lengths = []
    sentences = []
    labels = []
    unknown = 4
    seg_ch = [self.dictionary['。'], self.dictionary['！'], self.dictionary['？']]
    no_seg_ch = [self.dictionary['”']]
    characters_index = self.characters_index.tolist()
    labels_index = self.labels_index.tolist()
    line_lengths = list(map(lambda chs: len(chs), characters_index))

    def is_seg(item):
      return item[1] in seg_ch and (item[0] < item[2] - 1 and characters[item[0] + 1] not in no_seg_ch)

    for characters, label, length in zip(characters_index, labels_index, line_lengths):
      if length <= 1:
        continue
      seg_indices = [0] + [i[0] + 1 for i in filter(is_seg, zip(range(length), characters, [length] * length))]
      for pre_seg_index, cur_seg_index in zip(seg_indices[:-1], seg_indices[1:]):
        sentence = characters[pre_seg_index:cur_seg_index]
        sentence_labels = label[pre_seg_index:cur_seg_index]
        sentences.append(sentence)
        labels.append(sentence_labels)
        sentence_length = len(sentence)
        sentence_lengths.append(sentence_length)
        if sentence_length <= self.batch_length:
          pad_length = self.batch_length - sentence_length
          sentence_batches.append(sentence + [self.dictionary['BATCH_PAD']] * pad_length)
          label_batches.append(sentence_labels + [unknown] * pad_length)
          lengths.append(sentence_length)
        else:
          if sentence_labels[sentence_length - 1] != 0 and 1 in sentence_labels[:sentence_length:-1]:
            last_index = sentence_labels[:sentence_length:-1].index(1)
            pad_length = self.batch_length - last_index
            sentence_batches.append(sentence[:last_index] + [self.dictionary['BATCH_PAD']] * pad_length)
            label_batches.append(sentence_labels[:last_index] + [unknown] * pad_length)
            lengths.append(last_index)
          else:
            sentence_batches.append(sentence[:self.batch_length])
            label_batches.append(sentence_labels[:self.batch_length])
            lengths.append(self.batch_length)

    extra_count = len(sentence_batches) % self.batch_size
    sentence_batches = np.array(sentence_batches[:-extra_count], dtype=np.int32).reshape(
      [-1, self.batch_size, self.batch_length])
    label_batches = np.array(label_batches[:-extra_count], dtype=np.int32).reshape(
      [-1, self.batch_size, self.batch_length])
    lengths = np.array(lengths[:-extra_count], dtype=np.int32).reshape([-1, self.batch_size])
    return sentence_batches, label_batches, lengths, sentences, labels, sentence_lengths


if __name__ == '__main__':
  # PrepareData(4600, 'pku', mode=TrainMode.Batch)
  # PrepareData(4000, 'pku', type=CorpusType.Test, dict_path='corpus/pku_dict.utf8')
  # PrepareData(5000, 'msr', mode=TrainMode.Batch)
  PrepareData(None, 'emr', dict_path='corpus/emr_dict.utf8',mode=TrainMode.Sentence)
