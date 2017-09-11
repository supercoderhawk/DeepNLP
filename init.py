# -*- coding: UTF-8 -*-
import os
from shutil import copyfile


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


if __name__ == '__main__':
  # extract_cws_file()
  # transfer_cws_file()
  merge_cws_file()
