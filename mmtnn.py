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
    self.vocab_size = 4000
    self.embed_size = 50
    self.concat_embed_size = self.window_size * self.embed_size
    self.learning_rate = 0.2

    pre = PreprocessData('pku',TrainMode.Batch)
    self.dictionary = pre.dictionary

    self.embeddings = tf.Variable(
      tf.truncated_normal([self.vocab_size, self.embed_size], stddev=1.0 / math.sqrt(self.embed_size),
                          dtype=self.dtype), name='embeddings')
    self.input = tf.placeholder(tf.int32,[None,2])
