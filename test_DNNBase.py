from unittest import TestCase
import numpy as np
from dnn_base import DNNBase

# -*- coding: UTF-8 -*-
class TestDNNBase(TestCase):
  def setUp(self):
    self.dnn_base = DNNBase()
  def test_viterbi(self):
    score = np.arange(10, 170, 10).reshape(4, 4).T
    A = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]])
    init_A = np.array([1, 1, 0, 0])
    labels = np.array([3, 3, 3, 3])
    current_path = self.dnn_base.viterbi(score, A, init_A)
    print(current_path)

  def test_viterbi_new(self):
    score = np.arange(10, 170, 10).reshape(4, 4).T
    A = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]])
    init_A = np.array([1, 1, 0, 0])
    labels = np.array([3,3,3,3])
    current_path = self.dnn_base.viterbi_new(score, A, init_A,labels)
    #print(current_path)
    #correct_path = np.array([1, 3, 1, 3])
    #correct_score = np.array([21, 102, 203, 364])
    #self.assertTrue(np.all(current_path == correct_path))
    #self.assertTrue(np.all(current_score == correct_score))

  def test_generate_transition_update(self):
    pass

  def test_generate_transition_update_index(self):
    pass

  def test_sentence2index(self):
    pass

  def test_index2seq(self):
    pass

  def test_tags2words(self):
    pass
