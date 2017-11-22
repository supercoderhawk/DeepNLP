# -*- coding: UTF-8 -*-
import os


def estimate_ner(current_labels, correct_labels):
  corr_dict = {}
  curr_dict = {}
  corr_start = -2
  curr_start = -2

  # print('curr',current_labels)
  # print('corr', correct_labels)
  for label_index, (curr_label, corr_label) in enumerate(zip(current_labels, correct_labels)):
    if corr_label == 1:
      corr_start = label_index
      if corr_start == label_index - 1:
        corr_dict[corr_start] = 1
    elif label_index > 0 and corr_label == 2 and correct_labels[label_index - 1] != 2:
      corr_dict[corr_start] = label_index - corr_start

    if curr_label == 1:
      curr_start = label_index
      if curr_start == label_index - 1:
        curr_dict[curr_start] = 1
    elif label_index > 0 and curr_label == 2 and current_labels[label_index - 1] != 2:
      curr_dict[curr_start] = label_index - curr_start

  corr_count = 0
  prec_length = len(curr_dict)
  recall_length = len(corr_dict)
  for curr_start in curr_dict:
    if curr_start in corr_dict and curr_dict[curr_start] == corr_dict[curr_start]:
      corr_count += 1

  return corr_count, prec_length,recall_length

def prepare_for_crfpp(folder, output_name):
  content = []
  filenames = set()
  for _, _, names in os.walk(folder):
    for filename in names:
      name, _ = os.path.splitext(filename)
      if name not in filenames:
        filenames.add(name)
  for filename in filenames:
    path = folder + filename
    with open(path + '.txt', encoding='utf-8') as src_file:
      raw_text = src_file.read().replace('\n', '\r\n')
      labels = len(raw_text) * ['O']
      with open(path + '.ann', encoding='utf-8') as ann_file:
        ann_items = ann_file.read().splitlines()
        for item in ann_items:
          sections = item.split('\t')
          if sections[0].startswith('T'):
            pos = sections[1].split(' ')
            start, end = int(pos[1]), int(pos[2])
            labels[start] = 'B'
            if end - start - 1 > 0:
              labels[start + 1:end] = ['I'] * (end - start - 1)
      for ch, l in zip(raw_text, labels):
        if ch == '\r':
          continue
        if ch == '。':
          content.append(ch + '\t' + l + '\n')
        else:
          content.append(ch + '\t' + l)
  with open(output_name, mode='w', encoding='utf-8') as o:
    o.write('\n'.join(content))


def evaluate_ner(path):
  with open(path, encoding='utf-8') as f:
    entries = map(lambda l: l.split('\t'), [l for l in f.read().splitlines() if l])
    res = list(zip(*entries))
    label_map = {'O': 0, 'B': 1, 'I': 2}
    correct = list(map(lambda l: label_map[l], res[1]))
    current = list(map(lambda l: label_map[l], res[2]))
    corr, p_count, r_count = estimate_ner(current, correct)
    p = corr / p_count
    r = corr / r_count
    f1 = 2 * p * r / (p + r)
    print('precision:', p)
    print('recall:', r)
    print('f1', f1)

if __name__ == '__main__':
  # train_folder = 'corpus/emr_paper/train/'
  # test_folder = 'corpus/emr_paper/test/'
  # prepare_for_crfpp(test_folder,'corpus/test.data')
  # prepare_for_crfpp(train_folder, 'corpus/train.data')
  # evaluate_ner('D:\Learning\master_project\clinicalText\CRF++-0.58\\res.data')
  evaluate_ner('D:\Learning\master_project\clinicalText\CRF++-0.58\\res_slim.data')
