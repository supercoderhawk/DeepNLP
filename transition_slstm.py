# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np

class TransitionSLSTM:
  def __init__(self):
    # 参数初始化
    self.dtype = tf.float32
    self.alpha = 0.2
    self.embed_size = 100
    self.hidden_unit = 50
    self.action_count = 5
    # 数据初始化

    # 构建模型
    self.sess = tf.Session()
    # placeholder
    self.stack = tf.placeholder(self.dtype, [self.embed_size, None])
    self.buffer = tf.placeholder(self.dtype, [self.embed_size, None])
    self.history_action = tf.placeholder(self.dtype, [self.embed_size, None])
    self.allowed_action = tf.placeholder(self.dtype, [self.embed_size, None])
    # 变量
    self.action = tf.Variable(tf.random_uniform([self.action_count, self.hidden_unit], -1, 1, dtype=self.dtype))
    self.action_bias = tf.Variable(tf.random_uniform([self.action_count, self.hidden_unit], -1, 1, dtype=self.dtype))

  def train_exe(self):
    pass

  def train_sentence(self, sentence, label):
    pass
