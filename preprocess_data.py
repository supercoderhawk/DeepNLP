# -*- coding: UTF-8 -*-
import numpy as np
from config import CorpusType


class PreprocessData:
  def __init__(self, corpus, type=CorpusType.Train):
    self.skip_window_left = 1
    self.skip_window_right = 1
    self.window_size = self.skip_window_left + self.skip_window_right + 1
    self.dict_path = 'corpus/' + corpus + '_dict.utf8'
    if type == CorpusType.Train:
      self.input_base = 'corpus/' + corpus + '_training'
    elif type == CorpusType.Test:
      self.input_base = 'corpus/' + corpus + '_test'
    self.characters = np.load(self.input_base + '_characters.npy')
    self.labels = np.load(self.input_base + '_labels.npy')
    self.characters_batch, self.labels_batch = self.generate_sentences_batch()
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

  def generate_sentences_batch(self):
    characters_batch = []
    labels_batch = []
    for i, sentence_words in enumerate(self.characters):
      if len(sentence_words) < max(self.skip_window_left, self.skip_window_right):
        continue
      extend_words = [1] * self.skip_window_left
      extend_words.extend(sentence_words)
      extend_words.extend([2] * self.skip_window_right)
      word_batch = list(
        map(lambda item: extend_words[item[0] - self.skip_window_left:item[0] + self.skip_window_right + 1],
            enumerate(extend_words[self.skip_window_left:-self.skip_window_right], self.skip_window_left)))
      characters_batch.append(np.array(word_batch, dtype=np.int32))
      labels_batch.append(np.array(self.labels[i], dtype=np.int32))

    return np.array(characters_batch), np.array(labels_batch)
