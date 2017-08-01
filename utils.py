#-*- coding: UTF-8 -*-
import matplotlib.pyplot as plt

def strQ2B(ustring):
  '''全角转半角'''
  rstring = ''
  for uchar in ustring:
    inside_code = ord(uchar)
    if inside_code == 12288:  # 全角空格直接转换
      inside_code = 32
    elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
      inside_code -= 65248
    rstring += chr(inside_code)
  return rstring


def plot_lengths( lengths):
  pre_i = lengths[0]
  count = []
  x = []
  j = 0
  for i in lengths:
    if pre_i == i:
      j += 1
    else:
      count.append(j)
      x.append(pre_i)
      j = 0
      pre_i = i

  print(len(list(filter(lambda l: l > 300, lengths))))
  print(len(lengths))
  x = range(len(count))
  plt.plot(x, count)
  plt.ylabel('长度')
  plt.show()