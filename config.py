# -*- coding: UTF-8 -*-
from enum import Enum


class CorpusType(Enum):
  Train = 1
  Test = 2


class TrainMode(Enum):
  Sentence = 1
  Batch = 2
