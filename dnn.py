# -*- coding: UTF-8 -*-
import math

import tensorflow as tf


class DNN:
  def __init__(self):
    # 参数初始化
    self.dtype = tf.float32
    self.skip_window_left = 1
    self.skip_window_right = 1
    self.window_size = self.skip_window_left + self.skip_window_right + 1
    self.vocab_size = 4000
    self.embed_size = 100
    self.hidden_units = 150
    self.tags = [0, 1, 2, 3]
    self.tag_count = len(self.tags)
    self.concat_embed_size = self.window_size * self.embed_size
    self.alpha = 0.02
    self.lam = 0.001
    # 数据初始化
    self.word_batch = None
    self.label_batch = None
    self.dictionary = None
    # 模型定义和初始化
    self.sess = tf.Session()
    self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)
    self.x = tf.placeholder(self.dtype, shape=[1, None, self.concat_embed_size])
    self.embeddings = tf.Variable(
      tf.truncated_normal([self.vocab_size, self.embed_size], stddev=-1.0 / math.sqrt(self.embed_size),
                          dtype=self.dtype), dtype=self.dtype, name='embeddings')
    self.w = tf.Variable(
      tf.truncated_normal([self.tag_count, self.hidden_units], stddev=1.0 / math.sqrt(self.concat_embed_size),
                          dtype=self.dtype), dtype=self.dtype)
    self.b = tf.Variable(tf.zeros([self.tag_count, 1]), dtype=self.dtype)
    self.transition = tf.Variable(tf.random_uniform([self.tag_count, self.tag_count], -0.05, 0.05, dtype=self.dtype),
                                  dtype=self.dtype)
    self.transition_init = tf.Variable(
      tf.random_uniform([self.tag_count, self.tag_count], -0.05, 0.05, dtype=self.dtype), dtype=self.dtype)

  def train(self):
    pass

  def train_batch(self):
    pass
