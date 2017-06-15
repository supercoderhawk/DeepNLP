# -*- coding: UTF-8 -*-
from enum import Enum


class CorpusType(Enum):
  Train = 1
  Test = 2


class TrainMode(Enum):
  Sentence = 1
  Batch = 2


class BaseConfig:
  def __init__(self, learning_rate, vocab_size, embed_size, hidden_units):
    self.learning_rate = learning_rate
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.hidden_units = hidden_units


class DNNConfig(BaseConfig):
  def __init__(self, learning_rate, skip_left, skip_right, vocab_size, embed_size, hidden_units):
    BaseConfig.__init__(learning_rate, vocab_size, embed_size, hidden_units)
    self.skip_window_left = skip_left
    self.skip_window_right = skip_right
    self.window_size = self.skip_window_left + self.skip_window_right + 1
    self.concat_embed_size = self.embed_size * self.window_size
