# -*- coding: UTF-8 -*-
import math
import time
import tensorflow as tf
import numpy as np
from dnn_base import DNNBase
from preprocess_data import PreprocessData


class DNN(DNNBase):
  def __init__(self, type='mlp'):
    DNNBase.__init__(self)
    # 参数初始化
    self.dtype = tf.float32
    self.skip_window_left = 1
    self.skip_window_right = 1
    self.window_size = self.skip_window_left + self.skip_window_right + 1
    self.vocab_size = 4000
    self.embed_size = 100
    self.hidden_units = 150
    self.tags = [0, 1, 2, 3]
    self.tags_count = len(self.tags)
    self.concat_embed_size = self.window_size * self.embed_size
    self.alpha = 0.02
    self.lam = 0.0001
    # 数据初始化
    pre = PreprocessData('pku')
    self.characters_batch = pre.characters_batch
    self.labels_batch = pre.labels_batch
    self.dictionary = pre.dictionary
    # 模型定义和初始化
    self.sess = tf.Session()
    self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)
    self.embeddings = tf.Variable(
      tf.truncated_normal([self.vocab_size, self.embed_size], stddev=-1.0 / math.sqrt(self.embed_size),
                          dtype=self.dtype), name='embeddings')
    self.input = tf.placeholder(tf.int32, shape=[None, self.window_size])
    self.label_index_correct = tf.placeholder(tf.int32, shape=[None, 2])
    self.label_index_current = tf.placeholder(tf.int32, shape=[None, 2])
    self.w = tf.Variable(
      tf.truncated_normal([self.tags_count, self.hidden_units], stddev=1.0 / math.sqrt(self.concat_embed_size),
                          dtype=self.dtype), name='w')
    self.b = tf.Variable(tf.zeros([self.tags_count, 1],dtype=self.dtype), name='b')
    self.transition = tf.Variable(tf.random_uniform([self.tags_count, self.tags_count], -0.05, 0.05, dtype=self.dtype))
    self.transition_init = tf.Variable(tf.random_uniform([self.tags_count], -0.05, 0.05, dtype=self.dtype))
    self.transition_holder = tf.placeholder(self.dtype, shape=self.transition.get_shape())
    self.transition_init_holder = tf.placeholder(self.dtype, shape=self.transition_init.get_shape())
    self.update_transition = self.transition.assign(
      tf.add((1 - self.alpha * self.lam) * self.transition, self.alpha * self.transition_holder))
    self.update_transition_init = self.transition_init.assign(
      tf.add((1 - self.alpha * self.lam) * self.transition_init, self.alpha * self.transition_init_holder))
    self.params = [self.w, self.b, self.embeddings]
    if type == 'mlp':
      self.input_embeds = tf.transpose(tf.reshape(tf.nn.embedding_lookup(self.embeddings, self.input),
                                                  [-1, self.concat_embed_size]))
      self.hidden_w = tf.Variable(
        tf.random_uniform([self.hidden_units, self.concat_embed_size], -4.0 / math.sqrt(self.concat_embed_size),
                          4 / math.sqrt(self.concat_embed_size), dtype=self.dtype), name='hidden_w')
      self.hidden_b = tf.Variable(tf.zeros([self.hidden_units, 1], dtype=self.dtype), name='hidden_b')
      self.word_scores = tf.matmul(self.w,
                                   tf.sigmoid(tf.matmul(self.hidden_w, self.input_embeds) + self.hidden_b)) + self.b
      self.params += [self.hidden_w, self.hidden_b]
    elif type == 'lstm':
      self.input_embeds = tf.reshape(tf.nn.embedding_lookup(self.embeddings, self.input),
                                                  [-1,1, self.concat_embed_size])
      self.lstm = tf.contrib.rnn.LSTMCell(self.hidden_units)
      self.lstm_output, self.lstm_out_state = tf.nn.dynamic_rnn(self.lstm, self.input_embeds, dtype=self.dtype,
                                                                time_major=True)
      #tf.global_variables_initializer().run(session=self.sess)
      self.word_scores = tf.matmul(self.w, tf.transpose(self.lstm_output[:, -1, :])) + self.b
      self.params += [v for v in tf.global_variables() if v.name.startswith('rnn')]
    self.loss = tf.reduce_sum(
      tf.gather_nd(self.word_scores, self.label_index_current) -
      tf.gather_nd(self.word_scores, self.label_index_correct)) + tf.contrib.layers.apply_regularization(
      tf.contrib.layers.l2_regularizer(self.lam), self.params)
    self.train = self.optimizer.minimize(self.loss)
    self.saver = tf.train.Saver(self.params + [self.transition, self.transition_init], max_to_keep=100)

  def train_exe(self):
    tf.global_variables_initializer().run(session=self.sess)
    self.sess.graph.finalize()
    epoches = 10
    last_time = time.time()
    for i in range(epoches):
      print('epoch:%d'%i)
      for sentence_index, (sentence, labels) in enumerate(zip(self.characters_batch, self.labels_batch)):
        self.train_sentence(sentence, labels)
        if sentence_index > 0 and sentence_index % 1000 == 0:
          print(sentence_index)
          print(time.time() - last_time)
          last_time = time.time()
      self.saver.save(self.sess, 'tmp/lstm-model%d.ckpt' % i)

  def train_sentence(self, sentence, labels):
    scores = self.sess.run(self.word_scores, feed_dict={self.input: sentence})
    current_labels = self.viterbi(scores, self.transition.eval(session=self.sess),
                                  self.transition_init.eval(session=self.sess))
    diff_tags = np.subtract(labels, current_labels)
    update_index = np.where(diff_tags != 0)[0]
    update_length = len(update_index)

    if update_length == 0:
      return

    update_labels_pos = np.stack([labels[update_index], update_index], axis=-1)
    update_labels_neg = np.stack([current_labels[update_index], update_index], axis=-1)
    self.sess.run(self.train, feed_dict={self.input: sentence, self.label_index_current: update_labels_neg,
                                         self.label_index_correct: update_labels_pos})
    # 更新转移矩阵
    transition_update, transition_init_update, update_init = self.gen_update_A(labels, current_labels)
    self.sess.run(self.update_transition, feed_dict={self.transition_holder: transition_update})
    if update_init:
      self.sess.run(self.update_transition_init, feed_dict={self.transition_init_holder: transition_init_update})

  def train_batch(self,sentences_batch,labels_batch):
    pass

  def seg(self, sentence, model_path='tmp/mlp-model0.ckpt', debug=False):
    self.saver.restore(self.sess, model_path)
    seq = self.index2seq(self.sentence2index(sentence))
    sentence_scores = self.sess.run(self.word_scores, feed_dict={self.input: seq})
    transition_init = self.transition_init.eval(session=self.sess)
    transition = self.transition.eval(session=self.sess)
    if debug:
      print(transition)
      print(sentence_scores.T)
    current_labels = self.viterbi(sentence_scores, transition, transition_init)
    return self.tags2words(sentence, current_labels), current_labels


if __name__ == '__main__':
  # dnn = DNN()
  dnn = DNN('lstm')
  dnn.train_exe()
