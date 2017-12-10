#-*- coding: UTF-8 -*-
import re
import os

def read_single_file(ann_file,raw_file):
  with open(raw_file, encoding='utf-8') as r:
    sentence = r.read()
    rn_indices = [m.start() for m in re.finditer('\n',sentence)]
    spans_diff = {}
    if len(rn_indices):
      spans = zip([-1]+rn_indices,rn_indices+[len(sentence)+len(rn_indices)])
      for i,(before,curr) in enumerate(spans):
        spans_diff[(before+2,curr)] = i*2
    raw_sentence = sentence
    sentence = sentence.replace('\n','')

  #periods = [m.start() for m in re.finditer('。', sentence)]
  periods = []
  sentence_len = len(sentence)
  last = 0
  sentences = []
  for i,ch in enumerate(sentence):
    if ch =='。':
      if i<sentence_len-1 and sentence[i+1]=='”':
        pass
      else:
        periods.append(i)
        sentences.append(sentence[last:i+1])
        last = i+1
  if last!= len(sentence):
    sentences.append(sentence[last:sentence_len])
  period_spans = {}
  sentence_spans = {}
  # sentences = sentence.split('。')
  # if sentences[-1] == '':
  #   sentences = [s+'。' for s in sentences]
  # else:
  #   sentences = [s+'。' for s in sentences[:-1]]+[sentences[-1]]
  sentence_dict = {k:{'text':k} for k in sentences}

  if len(periods):
    for s, e in zip([-1] + periods, periods + [len(sentence)]):
      period_spans[(s + 1, e + 1)] = s + 1


  with open(ann_file, encoding='utf-8') as a:
    entries = map(lambda l:l.strip().split(' '),a.read().replace('\t',' ').splitlines())

    for entry in entries:
      id = entry[0]
      if id.startswith('T'):
        start = int(entry[2])
        end = int(entry[3])
        text = entry[4]
        if len(rn_indices):
          flag = False
          for s,e in spans_diff:
            if s <= start and end <= e:
              diff = spans_diff[(s,e)]
              start -= diff
              end -= diff
              flag = True
              break
          if not flag:
            print('a fucked world')
        if sentence[start:end] != text:
          # print('=========')
          # print(end - start)
          # print(id)
          # print(ann_file)
          # print(sentence[start:end])
          # print(text)
          # print('fuck world')
          continue


        if len(period_spans):
          for s,e in period_spans:
            if s<= start and end<= e:
              new_sentence = sentence[s:e]
              if new_sentence not in sentence_dict:
                print(ann_file)
                print('fuck aa')
              new_diff = period_spans[(s,e)]
              start -= new_diff
              end -= new_diff
              if new_sentence[start:end] != text:
                print('fuck')
              entity = {'id': id, 'start': start, 'length': end - start, 'text': text}
              entities = sentence_dict[new_sentence].get('entities')
              if entities is not None:
                entities.append(entity)
              else:
                sentence_dict[new_sentence]['entities'] = [entity]
              break
  for sentence in sentence_dict:
    labels = ['O'] * len(sentence)
    if sentence_dict[sentence].get('entities') is not None:
      for entity in sentence_dict[sentence]['entities']:
        start = entity['start']
        end = start + entity['length']
        labels[start] = 'B'
        if end -start > 1:
          labels[start+1:end] = ['I']*(end-start-1)
    sentence_dict[sentence]['label'] = labels
  return sentence_dict



def read_emr(directory,dest_file):
  files = set()
  for f in os.listdir(directory):
    files.add(os.path.splitext(os.path.split(f)[1])[0])
  sentences = []
  for f in files:
    sentences.extend(read_single_file(directory+f+'.ann',directory+f+'.txt').values())
  text = ''
  with open(dest_file, 'w',encoding='utf-8') as f:
    for sentence in sentences:
      text += '\n'.join([' '.join(l) for l in zip(sentence['text'],sentence['label'])])
      text += '\n\n'

    f.write(text)

if __name__ == '__main__':
  read_emr('corpus/emr_paper/train/','corpus/emr_paper/emr_training.conll')
  read_emr('corpus/emr_paper/test/', 'corpus/emr_paper/emr_test.conll')