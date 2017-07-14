# -*- coding: UTF-8 -*-
import numpy as np
from utils import strQ2B
from collections import Counter


class PrepareDataMSRNer:
  def __init__(self):
    self.labels_dict = {'O': 0, 'B': 1, 'I': 2}
    self.labels_count = len(self.labels_dict)
    self.ext_dict_path = ['corpus/msr_dict.utf8', 'corpus/pku_dict.utf8', 'corpus/emr_dict.utf8']
    self.dict_path = 'corpus/msr_ner_dict.utf8'
    self.corpus_path = 'corpus/msr_ner_training.utf8'
    self.words, self.labels = self.read_content()
    self.dictionary, self.reverse_dictionary = self.build_dictionary()
    self.characters, self.character_labels = self.build_dataset()
    np.save('corpus/msr_ner_training_characters', self.characters)
    np.save('corpus/msr_ner_training_labels', self.character_labels)

  def read_content(self):
    words = []
    labels = []
    with open(self.corpus_path, 'r', encoding='utf8') as corpus_file:
      sentences = corpus_file.read().splitlines()
      for sentence in sentences:
        word = []
        label = []
        sections = sentence.strip().split(' ')
        for section in sections:
          pair = section.split('/')
          word.append(strQ2B(pair[0]))
          label.append(pair[1])
        words.append(word)
        labels.append(label)
    return words, labels

  def build_dictionary(self):
    dictionary = {}
    characters = []
    for dict_path in self.ext_dict_path:
      d = self.read_dictionary(dict_path)
      characters.extend(d.keys())
    content = ''
    for line in self.words:
      for word in line:
        content += word
    characters.extend(list(Counter(content)))
    characters = list(
      filter(lambda ch: ch != 'UNK' and ch != 'STRT' and ch != 'END' and ch != 'BATCH_PAD', set(characters)))
    dictionary['BATCH_PAD'] = 0
    dictionary['UNK'] = 1
    dictionary['STRT'] = 2
    dictionary['END'] = 3
    for index, character in enumerate(characters, 4):
      dictionary[character] = index

    with open(self.dict_path, 'w', encoding='utf8') as dict_file:
      for character in dictionary:
        dict_file.write(character + ' ' + str(dictionary[character]) + '\n')
    return dictionary, dict(zip(dictionary.values(), dictionary.keys()))

  @staticmethod
  def read_dictionary(dict_path):
    dict_file = open(dict_path, 'r', encoding='utf-8')
    dict_content = dict_file.read().splitlines()
    dictionary = {}
    dict_arr = map(lambda item: item.split(' '), dict_content)
    for _, dict_item in enumerate(dict_arr):
      dictionary[dict_item[0]] = int(dict_item[1])
    dict_file.close()
    return dictionary

  def build_dataset(self):
    seg_punctuation = ['。', '？', '！']
    characters = []
    labels = []
    for line_word, line_label in zip(self.words, self.labels):
      line_characters = []
      line_labels = []
      for word, label in zip(line_word, line_label):
        for ch in word:
          if ch in self.dictionary:
            line_characters.append(self.dictionary[ch])
          else:
            line_characters.append(1)
        if label == 'o':
          line_labels.extend([self.labels_dict['O']] * len(word))
        else:
          line_labels.append(self.labels_dict['B'])
          line_labels.extend([self.labels_dict['I']] * (len(word) - 1))
        if word in seg_punctuation:
          characters.append(np.array(line_characters, np.int32))
          labels.append(np.array(line_labels, np.int32))
          line_characters = []
          line_labels = []
      if len(line_characters) != 0:
        characters.append(np.array(line_characters, np.int32))
        labels.append(np.array(line_labels, np.int32))
    return np.array(characters), np.array(labels)


if __name__ == '__main__':
  PrepareDataMSRNer()
