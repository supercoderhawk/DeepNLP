# -*- coding: UTF-8 -*-
import os
import numpy as np
from collections import OrderedDict
import pickle
from itertools import chain
from utils import plot_lengths


class PrepareDataNer():
  def __init__(self, entity_batch_length=225, relation_batch_length=85, entity_batch_size=10, relation_batch_size=50):
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
    with open('data/rel_pairs', 'rb') as pairs_file:
      self.relation_constraint = pickle.load(pairs_file)
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
    self.words_dict_path = 'corpus/emr_words_dict.utf8'
    self.entity_batch_length = entity_batch_length
    self.relation_batch_length = relation_batch_length
    self.entity_batch_size = entity_batch_size
    self.relation_batch_size = relation_batch_size
    for _, _, filenames in os.walk(self.base_folder):
      for filename in filenames:
        filename, _ = os.path.splitext(filename)
        if filename not in self.filenames:
          self.filenames.append(filename)
    self.annotations = self.read_annotation()
    self.dictionary, self.reverse_dictionary = self.build_dictionary()
    self.words_dictionary = self.build_words_dictionary()
    self.characters, self.entity_labels, self.relations, self.all_relations = self.build_dataset(True)
    # self.plot_words_sentences()
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
    self.train_relation_batches = self.build_relation_batch(self.relations)
    self.all_relation_batches = self.build_relation_batch(self.relations)
    with open('corpus/emr_relation_batches.rel', 'wb') as f:
      pickle.dump(self.train_relation_batches, f)
    with open('corpus/emr_all_relation_batches.rel', 'wb') as f:
      pickle.dump(self.all_relation_batches, f)

  def read_annotation(self):
    annotation = {}
    for filename in self.filenames:
      with open(self.base_folder + filename + '.txt', encoding='utf8') as raw_file:
        raw_text = raw_file.read().replace('\n', '\r\n')
      with open(self.base_folder + filename + '.ann', encoding='utf8') as annotation_file:
        results = annotation_file.read().replace('\t', ' ').splitlines()
        annotation_results = {'entity': {}, 'relations': [], 'entity_start': {}, 'cws': {}}

        for result in results:
          sections = result.split(' ')
          if sections[0][0] == 'T':
            entity = {'id': sections[0], 'category': sections[1], 'start': int(sections[2]), 'end': int(sections[3]),
                      'content': sections[4]}
            annotation_results['entity_start'][int(sections[2])] = {'id': sections[0]}
            annotation_results['entity'][sections[0]] = entity
          elif sections[0][0] == 'R':
            relation = {'id': sections[0], 'category': sections[1], 'primary': sections[2].split(':')[-1],
                        'secondary': sections[3].split(':')[-1]}
            annotation_results['relations'].append(relation)
        with open(self.base_folder + filename + '.cws', encoding='utf8') as cws_file:
          words = cws_file.read().strip().split('  ')
          lengths = [0]

          for i, w in enumerate(words):
            lengths.append(lengths[-1] + len(w))
            words[i] = words[i].replace('\n', '')

          # 验证
          for e in annotation_results['entity'].values():
            s = e['start']
            end = e['end']
            if s in lengths and end in lengths:
              if lengths.index(end) - lengths.index(s) != 1:
                print(filename)
                print(e)

          annotation_results['cws']['words'] = words
          annotation_results['cws']['words_index'] = lengths
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
      content += self.annotations[filename]['raw']
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

  def build_words_dictionary(self):
    words = set()
    words_dictionary = {'BATCH_PAD': 0, 'UNK': 1}
    for filename in self.annotations:
      for word in self.annotations[filename]['annotation']['cws']['words']:
        words.add(word)
    with open(self.words_dict_path, 'w', encoding='utf8') as dict_path:
      dict_path.write('BATCH_PAD 0\n')
      dict_path.write('UNK 1\n')
      for w in words:
        if len(w) > 0:
          words_dictionary[w] = len(words_dictionary)
          dict_path.write(w + ' ' + str(words_dictionary[w]) + '\n')

    return words_dictionary

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

  def build_dataset(self, is_entity_category=False, is_relation_category=False, is_negative_relation=True):
    rn = ['\r', '\n']
    seg = [self.dictionary['。']]
    seg_in_sentence = [self.dictionary[',']]
    word_seg = [self.words_dictionary['。']]
    word_seg_in_sentence = [self.words_dictionary[',']]
    characters_index = []
    entity_labels = []
    train_relations = []
    all_relations = []
    relations = {}
    pos = 0
    neg = 0
    all_neg = 0
    max_len = 0

    for filename in self.filenames:
      raw_text = self.annotations[filename]['raw']
      annotations = self.annotations[filename]['annotation']
      cws_list = annotations['cws']['words']
      cws_list_index = annotations['cws']['words_index']
      entity_start = annotations['entity_start']
      all_entities = annotations['entity']
      character_index = []
      word_index = []
      entity_label = [self.entity_tags['O']] * len(raw_text)
      rn_index = []
      relation = {}
      primary_entity = []
      seg_index = [0]  # 分隔符的字索引
      word_seg_index = [0]  # 分隔符的词索引

      for index, character in enumerate(raw_text):
        if character in rn:
          rn_index.append(index)
        elif character not in self.dictionary:
          character_index.append(1)
        else:
          character_index.append(self.dictionary[character])

      for index, word in enumerate(cws_list):
        if word not in self.words_dictionary:
          word_index.append(1)
        else:
          word_index.append(self.words_dictionary[word])

      for entity_annotation in annotations['entity'].values():
        start = entity_annotation['start']
        end = entity_annotation['end']
        content = entity_annotation['content']
        type = entity_annotation['category']
        if is_entity_category:
          entity_label[start] = self.entity_category_labels[self.entity_categories[type] + '_B']
          if len(content) > 1:
            entity_label[start + 1:end] = [self.entity_category_labels[self.entity_categories[type] + '_O']] * (
              end - start - 1)
          else:
            entity_label[start] = self.entity_tags['B']
            if len(content) > 1:
              entity_label[start + 1:end] = [self.entity_tags['I']] * (end - start - 1)

      for relation_annotation in annotations['relations']:
        id = relation_annotation['id']
        type = relation_annotation['category']
        primary = relation_annotation['primary']
        secondary = relation_annotation['secondary']
        relation[primary] = (secondary, type, id)
        primary_entity.append(primary)

      # 处理回车
      if len(rn_index) != 0:
        entity_label = [l[1] for l in filter(lambda ch_item: ch_item[0] not in rn_index, enumerate(entity_label))]

      # 分割
      doc_length = len(character_index)
      for index, ch_index in enumerate(character_index):
        if ch_index in seg:
          if index != doc_length - 1 and self.dictionary['”'] != character_index[index + 1]:
            seg_index.append(index + 1)
      if seg_index[-1] != doc_length:
        seg_index.append(doc_length)

      words_length = len(word_index)
      for i, w in enumerate(word_index):
        if w in word_seg:
          if i != words_length - 1 and self.words_dictionary['”'] != word_index[i + 1]:
            word_seg_index.append(i)
      if word_seg_index[-1] != words_length:
        word_seg_index.append(words_length)

      # 检验
      if len(seg_index) != len(word_seg_index):
        print(filename)
        print(len(seg_index) - len(word_seg_index))

      for sentence_index, (cur_index, latter_index, cur_word_index, latter_word_index) in enumerate(
          zip(seg_index[:-1], seg_index[1:],
              word_seg_index[:-1], word_seg_index[1:])):
        sentence_id = filename + '-' + str(sentence_index)
        # 寻找最长句子
        if max_len < latter_word_index - cur_word_index:
          max_len = latter_word_index - cur_word_index

        # 以句号分隔的句子中每个字的索引
        characters_index.append(np.array(character_index[cur_index:latter_index], dtype=np.int32))
        # 每个字对应的实体标签
        entity_labels.append(np.array(entity_label[cur_index:latter_index], dtype=np.int32))

        # 处理关系
        entity_dict = {}  # 每个句子中所有实体字典，键为实体id，值为实体在句子中的索引
        positive_relations = []  # 训练用关系的'hash'，primary_id < secondary_id
        current_relations = []  # 已添加的关系`hash`，防止无序关系添加两次
        sentence_word_index = []  # 句子中每个词在词典中的索引
        all_positive_relations = []  # 未处理的关系的hash

        for ii, i in enumerate(cws_list_index[cur_word_index:latter_word_index]):
          sentence_word_index.append(self.words_dictionary[cws_list[cws_list_index.index(i)]])
          if entity_start.get(i) != None:
            entity_dict[entity_start[i]['id']] = ii

        arr = np.arange(0, latter_word_index - cur_word_index) + self.relation_batch_length - 1  # 位置索引baseline
        for primary_id in [e for e in entity_dict if e in primary_entity]:
          secondary_id = relation[primary_id][0]
          type = relation[primary_id][1]
          if is_relation_category:
            relation_label = [0] * self.relation_category_label_count
            relation_label[self.relation_category_labels[type]] = 1
          else:
            relation_label = [0, 1]

          primary = entity_dict[primary_id]
          if entity_dict.get(secondary_id) is not None:
            secondary = entity_dict[secondary_id]
            # 无向
            if primary_id > secondary_id:
              positive_relations.append(secondary_id + ':' + primary_id)
            else:
              positive_relations.append(primary_id + ':' + secondary_id)
            all_positive_relations.append(primary_id + ':' + secondary_id)
            relation_item = {'sentence': np.array(word_index[cur_word_index:latter_word_index], dtype=np.int32),
                             'primary': arr - primary, 'secondary': arr - secondary,
                             'label': relation_label}
            # train_relations.append(relation_item)
            if relations.get(sentence_id) == None:
              relations[sentence_id] = [relation_item]
            else:
              relations[sentence_id].append(relation_item)

            all_relations.append(relation_item)

        pos += len(positive_relations)
        entities = list(entity_dict.keys())
        # 添加非关系，可认为是负采样
        distance = 5
        if is_negative_relation:
          for entity_i, entity in enumerate(entities):
            secondaries = []
            for s in entities[:entity_i] + entities[entity_i + 1:]:
              secondary_constraint = self.relation_constraint.get(all_entities[entity]['category'])
              if secondary_constraint == None or all_entities[s]['category'] not in secondary_constraint:
                continue

              if entity < s:
                first, second = entity, s
              else:
                first, second = s, entity

              first_index, second_index = entity_dict[entity], entity_dict[s]
              if first_index > second_index:
                first_index, second_index = second_index, first_index
              for i in sentence_word_index[first_index:second_index + 1]:
                if i in word_seg_index:
                  second_index = i - 1

              rel_hash = first + ':' + second
              if rel_hash not in positive_relations and rel_hash not in current_relations:
                if abs(entity_dict[first] - entity_dict[second]) < distance:
                  secondaries.append(s)
                  current_relations.append(rel_hash)

            all_secondaries = [s for s in entities[:entity_i] + entities[entity_i + 1:]
                               if entity + ':' + s not in all_positive_relations]
            primary_start = entity_dict[entity]
            neg += len(secondaries)
            all_neg += len(all_secondaries)

            for s in secondaries:
              if is_relation_category:
                relation_label = [0] * self.relation_category_label_count
                relation_label[self.relation_category_labels['NoRelation']] = 1
              else:
                relation_label = [1, 0]
              relation_item = {'sentence': np.array(word_index[cur_word_index:latter_word_index], dtype=np.int32),
                               'primary': arr - primary_start, 'secondary': arr - entity_dict[s],
                               'label': relation_label}
              # train_relations.append(relation_item)
              if relations.get(sentence_id) == None:
                relations[sentence_id] = [relation_item]
              else:
                relations[sentence_id].append(relation_item)
            for s in all_secondaries:
              if is_relation_category:
                relation_label = [0] * self.relation_category_label_count
                relation_label[self.relation_category_labels['NoRelation']] = 1
              else:
                relation_label = [1, 0]
              relation_item = {'sentence': np.array(word_index[cur_word_index:latter_word_index], dtype=np.int32),
                               'primary': arr - primary_start, 'secondary': arr - entity_dict[s],
                               'label': relation_label}
              all_relations.append(relation_item)

    print(neg / (pos + neg))
    print(all_neg / (pos + all_neg))
    train_relations = [r for rs in relations.values() for r in rs]
    for i, chs in enumerate(characters_index):
      sentence = ''
      for ch in chs:
        sentence += self.reverse_dictionary[ch]
    return np.array(characters_index), np.array(entity_labels), train_relations, all_relations

  def plot_words_sentences(self):
    lengths = list(map(lambda r: len(r['sentence']), self.relations))
    lengths.sort()
    plot_lengths(lengths)

  def build_entity_batch(self, category=False):
    characters = []
    labels = []
    for line_characters, line_labels in zip(self.characters, self.entity_labels):
      length = len(line_characters)
      if length >= self.entity_batch_length:
        characters.append(line_characters[:self.entity_batch_length])
        labels.append(line_labels[:self.entity_batch_length])
      else:
        characters.append(
          line_characters.tolist() + [self.dictionary['BATCH_PAD']] * (self.entity_batch_length - length))
        if category:
          labels.append(line_labels.tolist() + [self.entity_category_labels['P']] * (self.entity_batch_length - length))
        else:
          labels.append(line_labels.tolist() + [self.entity_tags['P']] * (self.entity_batch_length - length))
    extra_count = len(characters) % self.entity_batch_size
    characters = np.array(characters[:-extra_count], np.int32).reshape(
      [-1, self.entity_batch_size, self.entity_batch_length])
    labels = np.array(labels[:-extra_count], np.int32).reshape([-1, self.entity_batch_size, self.entity_batch_length])
    return characters, labels

  def build_relation_batch(self, relations):
    relation_batches = []
    sentence_batch = []
    primary_batch = []
    secondary_batch = []
    label_batch = []
    index = 0
    for relation in relations:
      sentence = relation['sentence'].tolist()
      if len(sentence) > self.relation_batch_length:
        sentence = sentence[:self.relation_batch_length]
      else:
        sentence.extend([self.dictionary['BATCH_PAD']] * (self.relation_batch_length - len(sentence)))
      primary = relation['primary'].tolist()
      if len(primary) > self.relation_batch_length:
        primary = primary[:self.relation_batch_length]
      else:
        primary.extend(range(primary[-1] + 1, primary[-1] + 1 + self.relation_batch_length - len(primary)))
      secondary = relation['secondary'].tolist()
      if len(secondary) > self.relation_batch_length:
        secondary = secondary[:self.relation_batch_length]
      else:
        secondary.extend(range(secondary[-1] + 1, secondary[-1] + 1 + self.relation_batch_length - len(secondary)))
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
