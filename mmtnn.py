# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import math
from dnn_base import DNNBase
from preprocess_data import PreprocessData
from config import TrainMode

class MMTNN(DNNBase):
  def __init__(self):
    DNNBase.__init__(self)
    self.dtype = tf.float32
    self.vocab_size = 4500
    self.embed_size = 50
    self.concat_embed_size = self.window_size * self.embed_size
    self.learning_rate = 0.2
    self.lam = 0.0002
    pre = PreprocessData('pku',TrainMode.Batch)
    self.dictionary = pre.dictionary

    self.embeddings = self.weight_variable([self.vocab_size, self.embed_size])
    self.input = tf.placeholder(tf.int32,[None,2])


  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=1.0/math.sqrt(shape[-1]), dtype=self.dtype)
    return tf.Variable(initial)

  def train(self):
    pass

  def seg(self):
    pass