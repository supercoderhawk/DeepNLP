# -*- coding: UTF-8 -*-
import os
from shutil import copyfile
import re


def extract_cws_file():
  source_folder = 'corpus/cws_unannotated/'
  dest_folder = 'corpus/emr/cws/'
  judge_folder = 'corpus/emr/'
  cws_ext_name = '.cws'
  files = set()
  for i in os.listdir(judge_folder):
    if os.path.isfile(judge_folder + i):
      files.add(os.path.splitext(i)[0])
  for i in files:
    copyfile(source_folder + i + cws_ext_name, dest_folder + i + cws_ext_name)


def transfer_cws_file():
  source_dir = 'corpus/emr/cws/'
  dest_folder = 'corpus/emr/'
  cws_files = set()
  cws_ext_name = '.cws'
  origin_files = set()
  for i in os.listdir(source_dir):
    cws_files.add(os.path.splitext(i)[0])

  for i in os.listdir(dest_folder):
    if os.path.isfile(dest_folder + i):
      origin_files.add(os.path.splitext(i)[0])
  print(len(cws_files) == len(origin_files))

  for i in cws_files:
    copyfile(source_dir + i + cws_ext_name, dest_folder + i + cws_ext_name)


def merge_cws_file():
  content = ''
  dest_folder = 'corpus/'
  judge_folder = 'corpus/emr/'
  cws_ext_name = '.cws'
  files = set()

  for i in os.listdir(judge_folder):
    if os.path.isfile(judge_folder + i):
      files.add(os.path.splitext(i)[0])

  for i in files:
    with open(judge_folder + i + cws_ext_name, 'r', encoding='utf8') as f:
      content += f.read().replace('\n', '') + '\n'
  with open(dest_folder + 'emr_training.utf8', 'w', encoding='utf8') as f:
    f.write(content)


def merge_emr():
  base_folder = 'corpus/admission-annotation/'
  ext_name = '.txt'
  files = set()
  sentences = []
  for i in os.listdir(base_folder):
    files.add(i[:i.index('.')])
  for i in files:
    with open(base_folder + i + ext_name, encoding='utf8') as f:
      content = f.read().replace('\n', '')
      index = [m.start() for m in re.finditer('。', content)]
      l = len(content)
      sentence = ''
      for beg, end in zip([-1] + index[:-1], index):
        if end - beg <= 1:
          continue
        if end != l - 1:
          if content[end + 1] != '”':
            sentences.append(sentence + content[beg + 1:end + 1])
            sentence = ''
          else:
            sentence = content[beg + 1:end + 1]
        else:
          sentences.append(sentence + content[beg + 1:end + 1])
  l = [len(l) for l in sentences]
  print(max(l), min(l))
  with open('corpus/emr.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(sentences))


if __name__ == '__main__':
  # extract_cws_file()
  # transfer_cws_file()
  # merge_cws_file()
  merge_emr()
