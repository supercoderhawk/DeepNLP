# -*- coding: UTF-8 -*-
import math
import time
import tensorflow as tf
import numpy as np
from dnn_base import DNNBase
from preprocess_data import PreprocessData
from config import TrainMode


class DNN(DNNBase):
  def __init__(self, type='mlp', batch_size=20, batch_length=300, mode=TrainMode.Batch, is_seg=False):
    DNNBase.__init__(self)
    # 参数初始化
    self.dtype = tf.float32
    self.skip_window_left = 0
    self.skip_window_right = 0
    self.window_size = self.skip_window_left + self.skip_window_right + 1
    self.vocab_size = 4000
    self.embed_size = 100
    self.hidden_units = 150
    self.tags = [0, 1, 2, 3]
    self.tags_count = len(self.tags)
    self.concat_embed_size = self.window_size * self.embed_size
    self.learning_rate = 0.2
    self.lam = 0.0001
    self.batch_length = batch_length
    self.batch_size = batch_size
    self.mode = mode
    self.type = type
    self.is_seg = is_seg
    self.dropout_rate = 0.2
    # 数据初始化
    pre = PreprocessData('pku', self.mode)
    self.character_batches = pre.character_batches
    self.label_batches = pre.label_batches
    self.dictionary = pre.dictionary
    # 模型定义和初始化
    self.sess = tf.Session()
    self.embeddings = tf.Variable(
      tf.truncated_normal([self.vocab_size, self.embed_size], stddev=5.0 / math.sqrt(self.embed_size),
                          dtype=self.dtype), name='embeddings')
    self.input = tf.placeholder(tf.int32, shape=[None, self.window_size])
    self.label_index_correct = tf.placeholder(tf.int32, shape=[None, 2])
    self.label_index_current = tf.placeholder(tf.int32, shape=[None, 2])
    self.w = tf.Variable(
      tf.truncated_normal([self.tags_count, self.hidden_units], stddev=1.0 / math.sqrt(self.concat_embed_size),
                          dtype=self.dtype), name='w')
    self.b = tf.Variable(tf.zeros([self.tags_count, 1], dtype=self.dtype), name='b')
    self.transition = tf.Variable(tf.random_uniform([self.tags_count, self.tags_count], -0.05, 0.05, dtype=self.dtype))
    self.transition_init = tf.Variable(tf.random_uniform([self.tags_count], -0.05, 0.05, dtype=self.dtype))
    self.transition_holder = tf.placeholder(self.dtype, shape=self.transition.get_shape())
    self.transition_init_holder = tf.placeholder(self.dtype, shape=self.transition_init.get_shape())
    #self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
    #self.optimizer = tf.train.MomentumOptimizer(0.01,0.9)
    #self.optimizer = tf.train.AdamOptimizer(0.0001)#,beta1=0.1,beta2=0.001)
    self.update_transition = self.transition.assign(
      tf.add((1 - self.learning_rate * self.lam) * self.transition,
             self.learning_rate * self.transition_holder))
    self.update_transition_init = self.transition_init.assign(
      tf.add((1 - self.learning_rate * self.lam) * self.transition_init,
             self.learning_rate * self.transition_init_holder))
    self.look_up = tf.reshape(tf.nn.embedding_lookup(self.embeddings, self.input), [-1, self.concat_embed_size])
    self.params = [self.w, self.b, self.embeddings]
    # self.params = [self.w, self.embeddings]
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
      self.loss = tf.reduce_sum(
        tf.gather_nd(self.word_scores, self.label_index_current) -
        tf.gather_nd(self.word_scores, self.label_index_correct)) + tf.contrib.layers.apply_regularization(
        tf.contrib.layers.l2_regularizer(self.lam), self.params)
    elif type == 'lstm':
      # self.lstm = tf.contrib.rnn.LSTMCell(self.hidden_units)
      self.lstm = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_units)
      self.b = tf.Variable(tf.zeros([self.tags_count, 1, 1], dtype=self.dtype), name='b')
      # self.lstm = tf.contrib.rnn.BasicRNNCell(self.hidden_units)
      # self.lstm = tf.contrib.rnn.GRUCell(self.hidden_units)
      if self.mode == TrainMode.Batch:
        if not self.is_seg:
          self.lengths = pre.lengths
          self.input = tf.placeholder(tf.int32, shape=[self.batch_size, self.batch_length, self.window_size])
          self.input_embeds = tf.reshape(tf.nn.embedding_lookup(self.embeddings, self.input),
                                         [self.batch_size, self.batch_length, self.concat_embed_size])
          # self.input_embeds = tf.layers.dropout(self.input_embeds,self.dropout_rate)
          self.lstm_output, self.lstm_out_state = tf.nn.dynamic_rnn(self.lstm, self.input_embeds, dtype=self.dtype)
          self.params += [v for v in tf.global_variables() if v.name.startswith('rnn')]
          self.word_scores = tf.tensordot(self.w, tf.transpose(self.lstm_output), [[1], [0]]) + self.b
          # tf.reshape(
          #  tf.tile(tf.squeeze(self.b), [self.batch_length * self.batch_size]),
          #  [self.tags_count, self.batch_length, self.batch_size])
          # self.w = tf.transpose(self.w)
          # self.b = tf.transpose(self.b)
          # self.word_scores = tf.transpose(
          #  tf.matmul(tf.reshape(self.lstm_output, [-1, self.hidden_units]), self.w) + self.b)
          # self.label_holder = tf.placeholder(tf.int32, shape=[self.batch_size, self.batch_length])
          self.label_index_correct = tf.placeholder(tf.int32, shape=[None, 3])
          self.label_index_current = tf.placeholder(tf.int32, shape=[None, 3])
          self.transition_correct_holder = tf.placeholder(tf.int32, [None, 2])
          self.transition_current_holder = tf.placeholder(tf.int32, [None, 2])
          self.transition_init_correct_holder = tf.placeholder(tf.int32,[None,1])
          self.transition_init_current_holder = tf.placeholder(tf.int32,[None,1])
          self.loss_scores = tf.reduce_sum(tf.gather_nd(self.word_scores, self.label_index_current) -
                                           tf.gather_nd(self.word_scores, self.label_index_correct)) + tf.reduce_sum(
            tf.gather_nd(self.transition, self.transition_current_holder) - tf.gather_nd(self.transition,
                                                                                         self.transition_correct_holder))
          self.loss_scores_with_init = self.loss_scores + tf.reduce_sum(
            tf.gather_nd(self.transition_init, self.transition_init_current_holder) - tf.gather_nd(self.transition_init,
                                                                                                   self.transition_init_correct_holder))
          self.regularization = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.lam),
                                                                       self.params + [self.transition])
          self.regularization_with_init = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(self.lam),self.params + [self.transition, self.transition_init])
          self.loss = self.loss_scores / self.batch_size + self.regularization
          self.loss_with_init = self.loss_scores_with_init / self.batch_size + self.regularization_with_init
        else:
          self.input_embeds = tf.reshape(tf.nn.embedding_lookup(self.embeddings, self.input),
                                         [1, -1, self.concat_embed_size])
          self.lstm_output, self.lstm_out_state = tf.nn.dynamic_rnn(self.lstm, self.input_embeds, dtype=self.dtype)
          self.word_scores = tf.matmul(self.w, tf.transpose(self.lstm_output[-1, :, :])) + self.b[:, :, -1]
    if self.is_seg == False:
      self.train = self.optimizer.minimize(self.loss)
    if self.is_seg == False and self.type == 'lstm':
      self.train_with_init = self.optimizer.minimize(self.loss_with_init)
    # self.saver = tf.train.Saver(self.params + [self.transition, self.transition_init], max_to_keep=100)
    self.saver = tf.train.Saver(max_to_keep=100)
    self.sentence_index = 0

  def train_exe(self):
    tf.global_variables_initializer().run(session=self.sess)
    self.sess.graph.finalize()
    epoches = 20
    last_time = time.time()
    if self.mode == TrainMode.Sentence:
      for i in range(epoches):
        print('epoch:%d' % i)
        for sentence_index, (sentence, labels) in enumerate(zip(self.character_batches, self.label_batches)):
          self.train_sentence(sentence, labels)
          self.sentence_index= sentence_index
          if sentence_index > 0 and sentence_index % 1000 == 0:
            print(sentence_index)
            print(time.time() - last_time)
            last_time = time.time()
        if self.type == 'mlp':
          self.saver.save(self.sess, 'tmp/mlp-model%d.ckpt' % i)
        elif self.type == 'lstm':
          self.saver.save(self.sess, 'tmp/lstm-model%d.ckpt' % i)
    elif self.mode == TrainMode.Batch:
      for i in range(epoches):
        self.step = i
        print('epoch:%d' % i)
        for batch_index, (character_batch, label_batch, lengths) in enumerate(
            zip(self.character_batches, self.label_batches, self.lengths)):
          self.train_batch(character_batch, label_batch, lengths)
          if batch_index > 0 and batch_index % 100 == 0:
            print(batch_index)
            print(time.time() - last_time)
            last_time = time.time()
        self.saver.save(self.sess, 'tmp/lstm-bmodel%d.ckpt' % i)

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
    feed_dict = {self.input: sentence, self.label_index_current: update_labels_neg,
                 self.label_index_correct: update_labels_pos}
    self.sess.run(self.train, feed_dict)

    # 更新转移矩阵
    transition_update, transition_init_update, update_init = self.generate_transition_update(labels, current_labels)
    self.sess.run(self.update_transition, feed_dict={self.transition_holder: transition_update})
    if update_init:
      self.sess.run(self.update_transition_init, feed_dict={self.transition_init_holder: transition_init_update})

  def train_batch(self, sentence_batches, label_batches, lengths):
    scores = self.sess.run(self.word_scores, feed_dict={self.input: sentence_batches})
    # scores = self.sess.run(self.word_scores, feed_dict={self.input: sentence_batches}).reshape(
    #  [-1, self.batch_length, self.batch_size])
    transition = self.transition.eval(session=self.sess)
    transition_init = self.transition_init.eval(session=self.sess)
    update_labels_pos = None
    update_labels_neg = None
    current_labels = []
    trans_pos_indices = []
    trans_neg_indices = []
    trans_init_pos_indices = []
    trans_init_neg_indices = []
    for i in range(self.batch_size):
      #current_label = self.viterbi(scores[:, :lengths[i], i], transition, transition_init,is_constraint=True)
      current_label = self.viterbi_new(scores[:, :lengths[i], i], transition, transition_init,label_batches[i, :lengths[i]])
      current_labels.append(current_label)
      #print(current_label)
      diff_tag = np.subtract(label_batches[i, :lengths[i]], current_label)
      update_index = np.where(diff_tag != 0)[0]
      update_length = len(update_index)
      if update_length == 0:
        continue
      update_label_pos = np.stack([label_batches[i, update_index], update_index, i * np.ones([update_length])], axis=-1)
      # update_label_pos = np.stack([label_batches[i, update_index], i * self.batch_length + update_index], axis=-1)
      update_label_neg = np.stack([current_label[update_index], update_index, i * np.ones([update_length])], axis=-1)
      # update_label_neg = np.stack([current_label[update_index], i * self.batch_length + update_index], axis=-1)
      if update_labels_pos is not None:
        np.concatenate((update_labels_pos, update_label_pos))
        np.concatenate((update_labels_neg, update_label_neg))
      else:
        update_labels_pos = update_label_pos
        update_labels_neg = update_label_neg
      trans_pos_index, trans_neg_index, trans_init_pos, trans_init_neg, update_init = self.generate_transition_update_index(
        label_batches[i, :lengths[i]], current_labels[i])
      trans_pos_indices.extend(trans_pos_index)
      trans_neg_indices.extend(trans_neg_index)
      if update_init:
        trans_init_pos_indices.append(trans_init_pos)
        trans_init_neg_indices.append(trans_init_neg)

    if update_labels_pos is not None and update_labels_neg is not None:
      feed_dict = {self.input: sentence_batches, self.label_index_current: update_labels_neg,
                   self.label_index_correct: update_labels_pos, self.transition_current_holder: trans_neg_indices,
                   self.transition_correct_holder: trans_pos_indices}
      self.sess.run(self.train, feed_dict)
      '''
      if len(trans_init_pos_indices) == 0:
        self.sess.run(self.train, feed_dict)
      else:
        feed_dict[self.transition_init_correct_holder]=trans_init_pos_indices
        feed_dict[self.transition_init_current_holder] = trans_init_neg_indices
        self.sess.run(self.train_with_init,feed_dict)
      '''
    # 更新转移矩阵
    '''
    transition_updates = None
    transition_init_updates = None
    for i in range(self.batch_size):
      transition_update, transition_init_update, update_init = self.generate_transition_update(
        label_batches[i, :lengths[i]], current_labels[i])
      if transition_updates is not None:
        transition_updates += transition_update
      else:
        transition_updates = transition_update
      if update_init:
        if transition_init_updates is not None:
          transition_init_updates += transition_init_update
        else:
          transition_init_updates = transition_init_update

    self.sess.run(self.update_transition, feed_dict={self.transition_holder: transition_updates / self.batch_size})
    if transition_init_updates is not None:
      feed_dict = {self.transition_init_holder: transition_init_updates / self.batch_size}
      self.sess.run(self.update_transition_init, feed_dict)
    '''

  def seg(self, sentence, model_path='tmp/mlp-model0.ckpt', debug=False):
    self.saver.restore(self.sess, model_path)
    seq = self.index2seq(self.sentence2index(sentence))
    sentence_scores = self.sess.run(self.word_scores, feed_dict={self.input: seq})
    transition_init = self.transition_init.eval(session=self.sess)
    transition = self.transition.eval(session=self.sess)
    if debug:
      print(transition)
      embeds = self.sess.run(self.look_up, feed_dict={self.input: seq})
      if self.type == 'lstm':
        output = self.sess.run(self.lstm_output, feed_dict={self.input: seq})
        print(output[-1, :, 10])
      print(sentence_scores.T)
      b = self.b.eval(session=self.sess)
      print(b)
      # print(self.params[2].eval(session=self.sess))
      print(self.transition_init.eval(session=self.sess))
      # print(self.b.eval(session=self.sess))
      # print(self.w.eval(session=self.sess))
    current_labels = self.viterbi_new(sentence_scores, transition, transition_init)
    return self.tags2words(sentence, current_labels), current_labels


if __name__ == '__main__':
  # dnn = DNN('mlp', mode=TrainMode.Sentence)
  dnn = DNN('lstm')
  dnn.train_exe()
