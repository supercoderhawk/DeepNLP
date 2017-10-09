# -*- coding: UTF-8 -*-
import numpy as np
import os
from config import CorpusType, TrainMode


class PreprocessData:
  def __init__(self, corpus, mode, type=CorpusType.Train,force_generate=True):
    self.skip_window_left = 1
    self.skip_window_right = 1
    self.window_size = self.skip_window_left + self.skip_window_right + 1
    self.dict_path = 'corpus/' + corpus + '_dict.utf8'
    if type == CorpusType.Train:
      self.input_base = 'corpus/' + corpus + '_training'
    elif type == CorpusType.Test:
      self.input_base = 'corpus/' + corpus + '_test'
    if mode == TrainMode.Sentence:
      self.characters = np.load(self.input_base + '_characters.npy')
      self.labels = np.load(self.input_base + '_labels.npy')
      self.lengths = np.load(self.input_base + '_lengths.npy')
      self.character_batches, self.label_batches = self.generate_sentences()
    elif mode == TrainMode.Batch:
      self.characters = np.load(self.input_base + '_character_batches.npy')
      self.labels = np.load(self.input_base + '_label_batches.npy')
      self.lengths = np.load(self.input_base + '_lengths_batches.npy')
      self.output_base = 'corpus/dnn/' + corpus + '_training'
      self.ouput_suffix = '_' + str(self.skip_window_left) + '_' + str(self.skip_window_right)
      if os.path.exists(self.output_base + '_character_batches' + self.ouput_suffix + '.npy') and not force_generate:
        self.character_batches = np.load(self.output_base + '_character_batches' + self.ouput_suffix + '.npy')
        self.label_batches = np.load(self.output_base + '_label_batches' + self.ouput_suffix + '.npy')
      else:
        self.character_batches, self.label_batches = self.generate_batches()
        np.save(self.output_base + '_character_batches' + self.ouput_suffix, self.character_batches)
        np.save(self.output_base + '_label_batches' + self.ouput_suffix, self.label_batches)
    else:
      print('模式错误')
      exit(1)

    self.dictionary = self.read_dictionary()

  def read_dictionary(self):
    dict_file = open(self.dict_path, 'r', encoding='utf-8')
    dict_content = dict_file.read().splitlines()
    dictionary = {}
    dict_arr = map(lambda item: item.split(' '), dict_content)
    for _, dict_item in enumerate(dict_arr):
      dictionary[dict_item[0]] = int(dict_item[1])
    dict_file.close()

    return dictionary

  def generate_sentences(self):
    characters_batch = []
    labels_batch = []
    for i, sentence_words in enumerate(self.characters):
      if len(sentence_words) < max(self.skip_window_left, self.skip_window_right):
        continue
      extend_words = [2] * self.skip_window_left
      extend_words.extend(sentence_words)
      extend_words.extend([3] * self.skip_window_right)
      if self.skip_window_right == 0:
        et = enumerate(extend_words[self.skip_window_left:], self.skip_window_left)
      else:
        et = enumerate(extend_words[self.skip_window_left:-self.skip_window_right], self.skip_window_left)
      word_batch = list(
        map(lambda item: extend_words[item[0] - self.skip_window_left:item[0] + self.skip_window_right + 1],et))
      characters_batch.append(np.array(word_batch, dtype=np.int32))
      labels_batch.append(np.array(self.labels[i], dtype=np.int32))
    #print(characters_batch)
    return np.array(characters_batch), np.array(labels_batch)

  def generate_batches(self):
    character_batches = []
    label_batches = []
    for batch_index, batch in enumerate(self.characters):
      character_batch = []
      label_batch = []
      for sentence_index, sentence in enumerate(batch):
        extend_words = [2] * self.skip_window_left
        extend_words.extend(sentence)
        extend_words.extend([3] * self.skip_window_right)
        if self.skip_window_right != 0:
          word_batch = list(
            map(lambda item: extend_words[item[0] - self.skip_window_left:item[0] + self.skip_window_right + 1],
                enumerate(extend_words[self.skip_window_left:-self.skip_window_right], self.skip_window_left)))
        else:
          word_batch = list(
            map(lambda item: extend_words[item[0] - self.skip_window_left:item[0] + self.skip_window_right + 1],
                enumerate(extend_words[self.skip_window_left:], self.skip_window_left)))
        character_batch.append(word_batch)
        label_batch.append(self.labels[batch_index][sentence_index])
      character_batches.append(character_batch)
      label_batches.append(label_batch)

    return np.array(character_batches, dtype=np.int32), np.array(label_batches, dtype=np.int32)
