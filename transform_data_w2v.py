# -*- coding: UTF-8 -*-
import numpy as np
import collections
import random


class TransformDataW2V(object):
  def __init__(self, batch_size, skip_window, source_file='corpus/emr.txt', dict_path='corpus/emr_embed_dict.utf8'):
    self.dict_path = dict_path
    self.batch_size = batch_size
    self.skip_window = skip_window
    self.source_file = source_file
    self.dictionary = self.read_dictionary()
    self.span = 2 * self.skip_window + 1
    self.indices = self.get_indices('emr')
    self.input, self.output = self.generate_collections()
    self.size = len(self.input)
    self.start = 0

  def read_dictionary(self):
    dict_file = open(self.dict_path, 'r', encoding='utf-8')
    dict_content = dict_file.read().splitlines()
    dictionary = {}
    dict_arr = map(lambda item: item.split(' '), dict_content)
    for _, dict_item in enumerate(dict_arr):
      dictionary[dict_item[0]] = int(dict_item[1])
    dict_file.close()

    return dictionary

  def build_dictionary(self, source_file):
    dictionary = {'UNK': 0}
    with open(source_file, 'r', encoding='utf8') as f:
      characters = set(''.join(f.readlines()))
    for i, ch in enumerate(characters):
      dictionary[ch] = i + 1
    with open(self.dict_path, 'w', encoding='utf8') as dict_file:
      for character in dictionary:
        dict_file.write(character + ' ' + str(dictionary[character]) + '\n')
    return dictionary

  def get_indices(self, name):
    lines = []
    with open('corpus/' + name + '.txt', 'r', encoding='utf8') as file:
      sentences = file.readlines()
      for sentence in sentences:
        if sentence:
          lines.append(self.sentence2index(sentence))

    return lines

  def sentence2index(self, sentence):
    index = []
    for ch in sentence:
      if ch in self.dictionary:
        index.append(self.dictionary[ch])
      else:
        index.append(self.dictionary['UNK'])
    return index

  def generate_collections(self):
    input = []
    output = []
    for index in self.indices:
      target = index[self.skip_window:-self.skip_window]
      input += [i for l in zip(*[target] * self.skip_window * 2) for i in l]
      target_index = range(self.skip_window, len(target) + self.skip_window)

      def shuffle(i):
        return random.sample(index[i - self.skip_window:i] + index[i + 1:i + self.skip_window + 1],
                             2 * self.skip_window)

      output += [j for i in map(shuffle, target_index) for j in i]
      if len(input) != len(output):
        print(len(input) - len(output))

    return input, output

  def generate_batch(self):
    if self.start + self.batch_size > self.size:
      input_batch = self.input[self.start:] + self.input[:self.batch_size + self.start - self.size]
      output_batch = self.output[self.start:] + self.output[:self.batch_size + self.start - self.size]
    else:
      input_batch = self.input[self.start:self.start + self.batch_size]
      output_batch = self.output[self.start:self.start + self.batch_size]

    self.start += self.batch_size
    self.start %= self.size

    return np.array(input_batch, dtype=np.int32), np.expand_dims(np.array(output_batch, dtype=np.int32), 1)
