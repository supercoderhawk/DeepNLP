# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import pickle


class RECNN():
  def __init__(self, batch_size=50, batch_length=225):
    self.dtype = tf.float32
    self.window_size = 3
    self.filter_size = 150
    self.relation_count = 2
    self.batch_length = batch_length
    self.batch_size = batch_size
    self.learning_rate = 0.01
    self.character_embed_size = 50
    self.position_embed_size = 50
    self.dict_path = 'corpus/emr_dict.utf8'
    self.dictionary = self.read_dictionary()
    self.character_size = len(self.dictionary)
    self.batch_path = 'corpus/emr_relation_batches.rel'
    self.concat_embed_size = self.character_embed_size + 2 * self.position_embed_size
    self.batches = self.load_batches()
    self.input_characters = tf.placeholder(tf.int32, [self.batch_size, self.batch_length])
    self.input_position = tf.placeholder(tf.int32, [self.batch_size, self.batch_length])
    self.input = tf.placeholder(self.dtype, [self.batch_size, self.batch_length, self.concat_embed_size, 1])
    self.input_relation = tf.placeholder(self.dtype, [self.batch_size, self.relation_count])
    self.position_embedding = self.weight_variable([2 * self.batch_length - 1, self.position_embed_size])
    self.character_embedding = self.weight_variable([self.character_size, self.character_embed_size])
    self.conv_kernel = self.weight_variable([self.window_size, self.concat_embed_size, 1, self.filter_size])
    self.bias = tf.Variable(0.1, dtype=self.dtype)
    self.full_connected_weight = self.weight_variable([self.filter_size, self.relation_count])
    self.position_lookup = tf.nn.embedding_lookup(self.position_embedding, self.input_position)
    self.character_lookup = tf.nn.embedding_lookup(self.character_embedding, self.input_characters)
    self.character_embed_holder = tf.placeholder(self.dtype,
                                                 [self.batch_size, self.batch_length, self.character_embed_size])
    self.primary_embed_holder = tf.placeholder(self.dtype,
                                               [self.batch_size, self.batch_length, self.position_embed_size])
    self.secondary_embed_holder = tf.placeholder(self.dtype,
                                                 [self.batch_size, self.batch_length, self.position_embed_size])
    self.emebd_concat = tf.expand_dims(
      tf.concat([self.character_embed_holder, self.primary_embed_holder, self.secondary_embed_holder], 2), 3)
    self.hidden_layer = tf.squeeze(self.max_pooling(tf.nn.relu(self.conv() + self.bias)))
    self.output = tf.nn.softmax(tf.matmul(self.hidden_layer, self.full_connected_weight))
    self.loss = tf.reduce_sum(tf.square(self.output - self.input_relation))
    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.train_model = self.optimizer.minimize(self.loss)

  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=self.dtype)
    return tf.Variable(initial)

  def conv(self):
    return tf.nn.conv2d(self.input, self.conv_kernel, strides=[1, 1, 1, 1], padding='VALID')

  def max_pooling(self, x):
    return tf.nn.max_pool(x, ksize=[1, self.batch_length - self.window_size + 1, 1, 1],
                          strides=[1, 1, 1, 1], padding='VALID')

  def train(self):
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      sess.graph.finalize()
      epoches = 10
      for i in range(epoches):
        print('epoch:' + str(i))
        for batch in self.batches:
          character_embeds, primary_embeds = sess.run([self.character_lookup, self.position_lookup],
                                                      feed_dict={self.input_characters: batch['sentence'],
                                                                 self.input_position: batch['primary']})
          secondary_embeds = sess.run(self.position_lookup, feed_dict={self.input_position: batch['secondary']})
          input = sess.run(self.emebd_concat, feed_dict={self.character_embed_holder: character_embeds,
                                                         self.primary_embed_holder: primary_embeds,
                                                         self.secondary_embed_holder: secondary_embeds})
          sess.run(self.train_model, feed_dict={self.input: input, self.input_relation: batch['label']})

  def load_batches(self):
    with open(self.batch_path, 'rb') as f:
      batches = pickle.load(f)
      return batches

  def read_dictionary(self):
    dict_file = open(self.dict_path, 'r', encoding='utf-8')
    dict_content = dict_file.read().splitlines()
    dictionary = {}
    dict_arr = map(lambda item: item.split(' '), dict_content)
    for _, dict_item in enumerate(dict_arr):
      dictionary[dict_item[0]] = int(dict_item[1])
    dict_file.close()
    return dictionary


if __name__ == '__main__':
  reCNN = RECNN()
  reCNN.train()
