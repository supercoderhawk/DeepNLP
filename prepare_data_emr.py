# -*- coding: UTF-8 -*-
import os
import numpy as np
from collections import OrderedDict
import pickle


class PrepareDataNer():
  def __init__(self, batch_length=225, entity_batch_size=10, relation_batch_size=50):
    self.entity_tags = {'O': 0, 'B': 1, 'I': 2, 'P': 3}
    self.entity_categories = {'Sign': 'SN', 'Symptom': 'SYM', 'Part': 'PT', 'Property': 'PTY', 'Degree': 'DEG',
                              'Quality': 'QLY', 'Quantity': 'QNY', 'Unit': 'UNT', 'Time': 'T', 'Date': 'DT',
                              'Result': 'RES',
                              'Disease': 'DIS', 'DiseaseType': 'DIT', 'Examination': 'EXN', 'Location': 'LOC',
                              'Medicine': 'MED', 'Spec': 'SPEC', 'Usage': 'USG', 'Dose': 'DSE', 'Treatment': 'TRT',
                              'Family': 'FAM',
                              'Modifier': 'MOF'}
    self.entity_category_labels = OrderedDict({'O': 0})
    entity_category_index = 1
    for category in self.entity_categories:
      self.entity_category_labels[self.entity_categories[category] + '_B'] = entity_category_index
      entity_category_index += 1
      self.entity_category_labels[self.entity_categories[category] + '_O'] = entity_category_index
      entity_category_index += 1
    self.entity_category_labels['P'] = entity_category_index
    self.entity_labels_count = len(self.entity_tags)
    self.relation_categories = {'PartOf': '部位', 'PropertyOf': '性质', 'DegreeOf': '程度', 'QualityValue': '定性值',
                                'QuantityValue': '定量值', 'UnitOf': '单位', 'TimeOf': '持续时间', 'StartTime': '开始时间',
                                'EndTime': '结束时间', 'Moment': '时间点', 'DateOf': '日期', 'ResultOf': '结果',
                                'LocationOf': '地点', 'DiseaseTypeOf': '疾病分型分期', 'SpecOf': '规格', 'UsageOf': '用法',
                                'DoseOf': '用量', 'FamilyOf': '家族成员', 'ModifierOf': '其他修饰词', 'UseMedicine': '用药',
                                'LeadTo': '导致', 'Find': '发现', 'Confirm': '证实', 'Adopt': '采取', 'Take': '用药',
                                'Limit': '限定', 'AlongWith': '伴随', 'Complement': '补足'}
    self.relation_category_labels = {'NoRelation': 0}
    relation_category_index = 1
    for relation_category in self.relation_categories:
      self.relation_category_labels[relation_category] = relation_category_index
      relation_category_index += 1
    self.relation_category_label_count = len(self.relation_category_labels)
    self.relation_labels = {'Y': 1, 'N': 0}
    self.relation_label_count = len(self.relation_labels)
    self.base_folder = 'corpus/emr/'
    self.filenames = []
    self.ext_dict_path = ['corpus/msr_dict.utf8', 'corpus/pku_dict.utf8']
    self.dict_path = 'corpus/emr_dict.utf8'
    self.batch_length = batch_length
    self.entity_batch_size = entity_batch_size
    self.relation_batch_size = relation_batch_size
    for _, _, filenames in os.walk(self.base_folder):
      for filename in filenames:
        filename, _ = os.path.splitext(filename)
        if filename not in self.filenames:
          self.filenames.append(filename)
    self.annotation = self.read_annotation()
    self.dictionary, self.reverse_dictionary = self.build_dictionary()
    self.characters, self.entity_labels, self.relations = self.build_dataset(True)
    np.save('corpus/emr_training_characters', self.characters)
    np.save('corpus/emr_training_labels', self.entity_labels)
    with open('corpus/emr_relations.rel', 'wb') as f:
      pickle.dump(self.relations, f)
    extra_count = len(self.characters) % self.entity_batch_size
    lengths = np.array(list(map(lambda item: len(item), self.characters[:-extra_count])), np.int32).reshape(
      [-1, self.entity_batch_size])
    np.save('corpus/emr_training_lengths', lengths)
    self.character_batches, self.label_batches = self.build_entity_batch()
    np.save('corpus/emr_training_character_batches', self.character_batches)
    np.save('corpus/emr_training_label_batches', self.label_batches)
    self.relation_batches = self.build_relation_batch()
    with open('corpus/emr_relation_batches.rel', 'wb') as f:
      pickle.dump(self.relation_batches, f)

  def read_annotation(self):
    annotation = {}
    for filename in self.filenames:
      with open(self.base_folder + filename + '.txt', encoding='utf8') as raw_file:
        raw_text = raw_file.read().replace('\n', '\r\n')
      with open(self.base_folder + filename + '.ann', encoding='utf8') as annotation_file:
        results = annotation_file.read().replace('\t', ' ').splitlines()
        annotation_results = {'entity': [], 'relation': []}
        for result in results:
          sections = result.split(' ')
          if sections[0][0] == 'T':
            entity = {'id': sections[0], 'category': sections[1], 'start': int(sections[2]), 'end': int(sections[3]),
                      'content': sections[4]}
            annotation_results['entity'].append(entity)
          elif sections[0][0] == 'R':
            relation = {'id': sections[0], 'category': sections[1], 'primary': sections[2].split(':')[-1],
                        'secondary': sections[3].split(':')[-1]}
            annotation_results['relation'].append(relation)
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
    # print(len(list(content)) / 1024)
    characters.extend(list(content.replace('\r\n', '')))
    characters = list(
      filter(lambda ch: ch != 'UNK' and ch != 'STRT' and ch != 'END' and ch != 'BATCH_PAD', set(characters)))
    dictionary['BATCH_PAD'] = 0
    dictionary['UNK'] = 1
    dictionary['STRT'] = 2
    dictionary['END'] = 3
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

  def build_dataset(self, entity_category=False, relation_category=False):
    rn = ['\r', '\n']
    seg = [self.dictionary['。']]
    characters_index = []
    entity_labels = []
    relations = []

    for filename in self.filenames:
      raw_text = self.annotation[filename]['raw']
      annotations = self.annotation[filename]['annotation']
      character_index = []
      entity_label = [self.entity_tags['O']] * len(raw_text)
      rn_index = []
      entity_index = [0] * len(raw_text)
      relation = {}
      primary_entity = []
      seg_index = [0]

      for index, character in enumerate(list(raw_text)):
        if character in rn:
          rn_index.append(index)
        elif character not in self.dictionary:
          character_index.append(1)
        else:
          character_index.append(self.dictionary[character])

      for entity_annotation in annotations['entity']:
        start = entity_annotation['start']
        end = entity_annotation['end']
        content = entity_annotation['content']
        type = entity_annotation['category']
        entity_index[start:end] = [entity_annotation['id']] * (end - start)
        if entity_category:
          entity_label[start] = self.entity_category_labels[self.entity_categories[type] + '_B']
          if len(content) > 1:
            entity_label[start + 1:end] = [self.entity_category_labels[self.entity_categories[type] + '_O']] * (
              end - start - 1)
          else:
            entity_label[start] = self.entity_tags['B']
            if len(content) > 1:
              entity_label[start + 1:end] = [self.entity_tags['I']] * (end - start - 1)
      for relation_annotation in annotations['relation']:
        id = relation_annotation['id']
        type = relation_annotation['category']
        primary = relation_annotation['primary']
        secondary = relation_annotation['secondary']
        relation[primary] = (secondary, type, id)
        primary_entity.append(primary)

      # 处理回车
      if len(rn_index) != 0:
        entity_label = [l[1] for l in filter(lambda ch_item: ch_item[0] not in rn_index, enumerate(entity_label))]
        entity_index = [l[1] for l in filter(lambda ch_item: ch_item[0] not in rn_index, enumerate(entity_index))]

      # 分割
      doc_length = len(character_index)
      for index, ch_index in enumerate(character_index):
        if ch_index in seg:
          if index != doc_length - 1 and self.dictionary['”'] != character_index[index + 1]:
            seg_index.append(index + 1)
      if seg_index[-1] != len(character_index):
        seg_index.append(len(character_index))

      for cur_index, latter_index in zip(seg_index[:-1], seg_index[1:]):
        characters_index.append(np.array(character_index[cur_index:latter_index], dtype=np.int32))
        entity_labels.append(np.array(entity_label[cur_index:latter_index], dtype=np.int32))
        entity_start = {}
        for ii, i in enumerate(entity_index[cur_index:latter_index]):
          if i != 0 and entity_start.get(i) is None:
            entity_start[i] = ii
        for entity_id in [e for e in entity_start if e in primary_entity]:
          secondary = relation[entity_id][0]
          type = relation[entity_id][1]
          if relation_category:
            relation_label = [0] * self.relation_category_label_count
            relation_label[self.relation_category_labels[type]] = 1
          else:
            relation_label = [0, 1]

          primary_start = entity_start[entity_id]
          if entity_start.get(secondary) is not None:
            secondary_start = entity_start[secondary]
            arr = np.arange(0, latter_index - cur_index) + self.batch_length - 1
            relation_item = {'sentence': np.array(character_index[cur_index:latter_index], dtype=np.int32),
                             'primary': arr - primary_start, 'secondary': arr - secondary_start,
                             'label': relation_label}
            relations.append(relation_item)

    for i, chs in enumerate(characters_index):
      sentence = ''
      for ch in chs:
        sentence += self.reverse_dictionary[ch]
    return np.array(characters_index), np.array(entity_labels), relations

  def build_entity_batch(self, category=False):
    characters = []
    labels = []
    for line_characters, line_labels in zip(self.characters, self.entity_labels):
      length = len(line_characters)
      if length >= self.batch_length:
        characters.append(line_characters[:self.batch_length])
        labels.append(line_labels[:self.batch_length])
      else:
        characters.append(line_characters.tolist() + [self.dictionary['BATCH_PAD']] * (self.batch_length - length))
        if category:
          labels.append(line_labels.tolist() + [self.entity_category_labels['P']] * (self.batch_length - length))
        else:
          labels.append(line_labels.tolist() + [self.entity_tags['P']] * (self.batch_length - length))
    extra_count = len(characters) % self.entity_batch_size
    characters = np.array(characters[:-extra_count], np.int32).reshape([-1, self.entity_batch_size, self.batch_length])
    labels = np.array(labels[:-extra_count], np.int32).reshape([-1, self.entity_batch_size, self.batch_length])
    return characters, labels

  def build_relation_batch(self):
    relation_batches = []
    sentence_batch = []
    primary_batch = []
    secondary_batch = []
    label_batch = []
    index = 0
    for relation in self.relations:
      sentence = relation['sentence'].tolist()
      if len(sentence) > self.batch_length:
        sentence = sentence[:self.batch_length]
      else:
        sentence.extend([self.dictionary['BATCH_PAD']] * (self.batch_length - len(sentence)))
      primary = relation['primary'].tolist()
      if len(primary) > self.batch_length:
        primary = primary[:self.batch_length]
      else:
        primary.extend(range(primary[-1] + 1, primary[-1] + 1 + self.batch_length - len(primary)))
      secondary = relation['secondary'].tolist()
      if len(secondary) > self.batch_length:
        secondary = secondary[:self.batch_length]
      else:
        secondary.extend(range(secondary[-1] + 1, secondary[-1] + 1 + self.batch_length - len(secondary)))
      sentence_batch.append(sentence)
      primary_batch.append(primary)
      secondary_batch.append(secondary)
      label_batch.append(relation['label'])
      index += 1
      if index > 0 and index % self.relation_batch_size == 0:
        batch = {'sentence': np.array(sentence_batch, np.int32), 'primary': np.array(primary_batch, np.int32),
                 'secondary': np.array(secondary_batch, np.int32), 'label': np.array(label_batch, np.float32)}
        relation_batches.append(batch)
        sentence_batch.clear()
        primary_batch.clear()
        secondary_batch.clear()
        label_batch.clear()
        index = 0
    return relation_batches


if __name__ == '__main__':
  PrepareDataNer()
