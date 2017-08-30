# -*- coding: UTF-8 -*-
import re
import numpy as np
import pickle
from functools import reduce

class PrepareDataSemeval:
  def __init__(self, batch_length=95, batch_size=50):
    self.base_path = 'corpus/semeval_relation'
    self.path = self.base_path + '.utf8'
    self.batch_length = batch_length
    self.batch_size = batch_size
    self.relation_categories, self.relations = self.read_content()
    self.dictionary = self.build_dictionary()
    self.batches = self.build_batches()
    with open('corpus/semeval_relation_batches.rel', 'wb') as f:
      pickle.dump(self.batches, f)
    print(len(self.relation_categories))

  def read_content(self):
    with open(self.path, 'r', encoding='utf8') as file:
      contents = file.read().split('\n\n')
      relation_categories = {'Other': 0}
      relation_count = {'Other': 0}
      length = []
      relations = []
      for content in contents:
        sections = content.splitlines()
        if len(sections) == 0:
          break

        idx = re.search(r'^[0-9]+', sections[0])
        if idx:
          idx = idx.group(0)
        else:
          idx = '1'

        sentence = sections[0][len(idx) + 2:-1]
        if idx == '1':
          sentence = sentence[1:]

        reduce_lengths = [4, 9, 13]
        e1_starttag_pos = sentence.find('<e1>')
        e1_endtag_pos = sentence.find('</e1>')
        e2_starttag_pos = sentence.find('<e2>')
        e2_endtag_pos = sentence.find('</e2>')
        e1_start = e1_starttag_pos
        e1_end = e1_endtag_pos - reduce_lengths[0]
        e2_start = e2_starttag_pos - reduce_lengths[1]
        e2_end = e2_endtag_pos - reduce_lengths[2]
        raw_sentence = sentence.replace('<e1>', '').replace('</e1>', '')\
                          .replace('<e2>', '').replace('</e2>','').strip()
        e1 = raw_sentence[e1_start:e1_end]
        e2 = raw_sentence[e2_start:e2_end]
        raw_words = list(filter(lambda w:len(w)>0 and w != ' ',raw_sentence.split(' ')))  # 带标点的单词
        raw_words_index = [0]
        for raw_word in raw_words[:-1]:
          raw_words_index.append(raw_words_index[-1] + len(raw_word) + 1)

        words = []  # 分离标点后的单词
        words_index = []
        for raw_word_index, raw_word in enumerate(raw_words):
          if len(raw_word) > 1:
            if len(raw_word) >2 and not raw_word[-1].isalnum() and not raw_word[0].isalnum():
              words.append(raw_word[0])
              words.append(raw_word[1:-1])
              words.append(raw_word[-1])
              words_index.append(raw_words_index[raw_word_index])
              words_index.append(raw_words_index[raw_word_index]+1)
              words_index.append(raw_words_index[raw_word_index] + len(raw_word) - 1)
            elif not raw_word[-1].isalnum():
              words.append(raw_word[:-1])
              words.append(raw_word[-1])
              words_index.append(raw_words_index[raw_word_index])
              words_index.append(raw_words_index[raw_word_index] + len(raw_word) - 1)
            elif not raw_word[0].isalnum():
              words.append(raw_word[0])
              words.append(raw_word[1:])
              words_index.append(raw_words_index[raw_word_index])
              words_index.append(raw_words_index[raw_word_index] + 1)
            else:
              words.append(raw_word)
              words_index.append(raw_words_index[raw_word_index])
        length.append(len(words))

        e1_index = words_index.index(e1_start)
        e2_index = words_index.index(e2_start)
        for word in words:
          if len(word) == 0:
            print('fuck')
        if sections[1] != 'Other':
          relation = re.search(r'([a-zA-Z-]*)\(', sections[1]).groups()[0]
          if relation not in relation_categories:
            relation_categories[relation] = len(relation_categories)
            relation_count[relation] = 0
          primary, secondary = re.search(r'\((\S+),(\S+)\)', sections[1]).groups()
          if primary == 'e2' and secondary == 'e1':
            e1_index = words_index.index(e2_start)
            e2_index = words_index.index(e1_start)
        else:
          relation = 'Other'
        relation_count[relation] += 1
        relations.append({'id': idx, 'words': words, 'primary': e1_index, 'secondary': e2_index,
                          'type': relation_categories[relation]})

      print(relation_count)
      print(relation_categories)
      all_count = reduce(lambda a,b:a+b,relation_count.values())
      return relation_categories, relations

  def build_dictionary(self):
    words_set = set()
    for relation in self.relations:
      words_set = words_set.union(set(relation['words']))
    dictionary = {'BATCH_PAD': 0, 'UNK': 1}
    with open('corpus/semeval_dict.utf8', 'w', encoding='utf8') as file:
      for word in words_set:
        dictionary[word] = len(dictionary)
        file.write(word+' '+str(dictionary[word])+'\n')
      file.write('BATCH_PAD'+' '+str(dictionary['BATCH_PAD'])+'\n')
      file.write('UNK' + ' ' + str(dictionary['UNK']) + '\n')
    return dictionary

  def build_batches(self):
    batches = []
    sentence = []
    primary = []
    secondary = []
    label = []
    base_index = range(self.batch_length, 2 * self.batch_length)
    for relation_index, relation in enumerate(self.relations,1):
      words = list(map(lambda w: self.dictionary[w], relation['words']))
      words += [self.dictionary['BATCH_PAD']] * (self.batch_length - len(words))
      sentence.append(words)
      base_index = range(self.batch_length,len(relation['words'])+self.batch_length)

      p = list(map(lambda i: i - relation['primary'], base_index))
      s = list(map(lambda i: i - relation['secondary'], base_index))

      p.extend([self.dictionary['BATCH_PAD']]*(self.batch_length-len(relation['words'])))
      s.extend([self.dictionary['BATCH_PAD']] * (self.batch_length - len(relation['words'])))

      primary.append(p)
      secondary.append(s)

      relation_arr = [0]*len(self.relation_categories)
      relation_arr[relation['type']] = 1
      label.append(relation_arr)

      if relation_index % self.batch_size == 0:
        batches.append({'sentence': np.array(sentence, np.int32), 'primary': np.array(primary, np.int32),
                        'secondary': np.array(secondary, np.int32), 'label': np.array(label, np.float32)})
        sentence.clear()
        primary.clear()
        secondary.clear()
        label.clear()

    return batches

if __name__ == '__main__':
  sem = PrepareDataSemeval()
