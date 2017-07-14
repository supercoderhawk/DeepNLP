# -*- coding: UTF-8 -*-
import os
import numpy as np


class PrepareDataNer():
  def __init__(self):
    self.labels_dict = {'O': 0, 'B': 1, 'I': 2}
    self.labels_count = len(self.labels_dict)
    self.base_folder = 'corpus/emr/'
    self.filenames = []
    self.ext_dict_path = ['corpus/msr_dict.utf8', 'corpus/pku_dict.utf8']
    self.dict_path = 'corpus/emr_dict.utf8'
    for _, _, filenames in os.walk(self.base_folder):
      for filename in filenames:
        filename, _ = os.path.splitext(filename)
        if filename not in self.filenames:
          self.filenames.append(filename)
    self.annotation = self.read_annotation_files()
    self.dictionary, self.reverse_dictionary = self.build_dictionary()
    self.character_batches, self.label_batches = self.build_dataset()
    np.save('corpus/emr_training_characters', self.character_batches)
    np.save('corpus/emr_training_labels', self.label_batches)

  def read_annotation_files(self):
    annotation = {}
    for filename in self.filenames:
      with open(self.base_folder + filename + '.txt', encoding='utf8') as raw_file:
        raw_text = raw_file.read().replace('\n', '\r\n')
      with open(self.base_folder + filename + '.ann', encoding='utf8') as annotation_file:
        annotation_results = annotation_file.read().splitlines()
      annotation[filename] = {'raw': raw_text, 'annotation': annotation_results}
    return annotation

  def build_dictionary(self):
    dictionary = {}
    characters = []
    for dict_path in self.ext_dict_path:
      d = self.read_dictionary(dict_path)
      characters.extend(d.keys())

    content = ''
    for filename in self.filenames:
      content += self.annotation[filename]['raw']
    print(len(list(content)) / 1024)
    characters.extend(list(content.replace('\r\n', '')))
    characters = list(filter(lambda ch: ch != 'UNK' and ch != 'STRT' and ch != 'END', set(characters)))
    dictionary['UNK'] = 0
    dictionary['STRT'] = 1
    dictionary['END'] = 2
    for index, character in enumerate(characters, 3):
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
    rn = ['\r', '\n']
    seg = [self.dictionary['。']]
    characters_index = []
    labels = []

    for filename in self.filenames:
      raw_text = self.annotation[filename]['raw']
      annotations = self.annotation[filename]['annotation']
      character_index = []
      label = [self.labels_dict['O']] * len(raw_text)
      rn_index = []
      seg_index = [0]

      for index, character in enumerate(list(raw_text)):
        if character in rn:
          rn_index.append(index)
        if character not in self.dictionary:
          character_index.append(1)
        else:
          character_index.append(self.dictionary[character])

      for annotation in annotations:
        annotation = annotation.replace('\t', ' ')
        if annotation[0] == 'T':
          sections = annotation.split(' ')
          start = int(sections[2])
          end = int(sections[3])
          if len(sections[4]) == 1:
            label[start] = self.labels_dict['B']
          elif len(sections[4]) > 1:
            label[start] = self.labels_dict['B']
            label[start + 1:end] = [self.labels_dict['I']] * (end - start - 1)

      # 处理回车
      if len(rn_index) != 0:
        character_index = list(map(lambda ch_item: ch_item[1],
                                   filter(lambda ch_item: ch_item[0] not in rn_index, enumerate(character_index))))
        label = list(
          map(lambda ch_item: ch_item[1], filter(lambda ch_item: ch_item[0] not in rn_index, enumerate(label))))

      doc_length = len(character_index)
      for index, ch_index in enumerate(character_index):
        if ch_index in seg:
          if index != doc_length - 1 and self.dictionary['”'] != character_index[index + 1]:
            seg_index.append(index + 1)
      if seg_index[-1] != len(character_index):
        seg_index.append(len(character_index))

      for cur_index, latter_index in zip(seg_index[:-1], seg_index[1:]):
        characters_index.append(np.array(character_index[cur_index:latter_index], dtype=np.int32))
        labels.append(np.array(label[cur_index:latter_index], dtype=np.int32))

        # characters_index.append(np.array(character_index,np.int32))
        # labels.append(np.array(label,np.int32))
    for i,chs in enumerate(characters_index):
      sentence = ''
      for ch in chs:
        sentence += self.reverse_dictionary[ch]
      print(sentence)
      print(labels[i])
    return np.array(characters_index), np.array(labels)

  def build_batch(self):
    pass

if __name__ == '__main__':
  PrepareDataNer()
