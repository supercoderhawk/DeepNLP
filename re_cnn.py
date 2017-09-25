# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import pickle


class RECNN():
  def __init__(self, relation_count=2, batch_size=50, batch_length=85):
    self.dtype = tf.float32
    self.window_size = 3
    self.filter_size = 150
    self.relation_count = relation_count
    self.batch_length = batch_length
    self.batch_size = batch_size
    self.learning_rate = 0.01
    self.dropout_rate = 0.5
    self.lam = 0.0005
    self.character_embed_size = 300
    self.position_embed_size = 50
    self.dict_path = 'corpus/emr_words_dict.utf8'
    self.dictionary = self.read_dictionary()
    self.words_size = len(self.dictionary)
    self.batch_path = 'corpus/emr_relation_batches.rel'
    self.test_batch_path = 'corpus/emr_test_relations.rel'
    self.output_folder = 'tmp/re/'
    self.concat_embed_size = self.character_embed_size + 2 * self.position_embed_size
    self.input_characters = tf.placeholder(tf.int32, [None, self.batch_length])
    self.input_position = tf.placeholder(tf.int32, [None, self.batch_length])
    self.input = tf.placeholder(self.dtype, [None, self.batch_length, self.concat_embed_size, 1])
    self.input_relation = tf.placeholder(self.dtype, [None, self.relation_count])
    self.position_embedding = self.weight_variable([2 * self.batch_length, self.position_embed_size])
    self.character_embedding = self.weight_variable([self.words_size, self.character_embed_size])
    self.conv_kernel = self.weight_variable([self.window_size, self.concat_embed_size, 1, self.filter_size])
    self.bias = self.weight_variable([self.filter_size])
    self.full_connected_weight = self.weight_variable([self.filter_size, self.relation_count])
    self.full_connected_bias = self.weight_variable([self.relation_count])
    self.position_lookup = tf.nn.embedding_lookup(self.position_embedding, self.input_position)
    self.character_lookup = tf.nn.embedding_lookup(self.character_embedding, self.input_characters)
    self.character_embed_holder = tf.placeholder(self.dtype,
                                                 [None, self.batch_length, self.character_embed_size])
    self.primary_embed_holder = tf.placeholder(self.dtype,
                                               [None, self.batch_length, self.position_embed_size])
    self.secondary_embed_holder = tf.placeholder(self.dtype,
                                                 [None, self.batch_length, self.position_embed_size])
    self.emebd_concat = tf.expand_dims(
      tf.concat([self.character_embed_holder, self.primary_embed_holder, self.secondary_embed_holder], 2), 3)
    self.hidden_layer = tf.layers.dropout(tf.squeeze(self.max_pooling(tf.nn.relu(self.conv() + self.bias))),
                                          self.dropout_rate)
    self.output_no_softmax = tf.matmul(self.hidden_layer, self.full_connected_weight) + self.full_connected_bias
    self.output = tf.nn.softmax(tf.matmul(self.hidden_layer, self.full_connected_weight) + self.full_connected_bias)
    self.params = [self.position_embedding, self.character_embedding, self.conv_kernel, self.bias,
                   self.full_connected_weight, self.full_connected_bias]
    self.regularization = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.lam),
                                                                 self.params)
    self.loss = tf.reduce_sum(tf.square(self.output - self.input_relation)) / self.batch_size + self.regularization
    self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_relation,
                                                                 logits=self.output_no_softmax) + self.regularization
    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    # self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
    self.train_model = self.optimizer.minimize(self.loss)
    self.train_cross_entropy_model = self.optimizer.minimize(self.cross_entropy)
    self.saver = tf.train.Saver(max_to_keep=100)

  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=self.dtype)
    return tf.Variable(initial)

  def conv(self):
    return tf.nn.conv2d(self.input, self.conv_kernel, strides=[1, 1, 1, 1], padding='VALID')

  def max_pooling(self, x):
    return tf.nn.max_pool(x, ksize=[1, self.batch_length - self.window_size + 1, 1, 1],
                          strides=[1, 1, 1, 1], padding='VALID')

  def train(self):
    batches = self.load_batches(self.batch_path)
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      sess.graph.finalize()
      epoches = 100
      for i in range(epoches):
        print('epoch:' + str(i))
        for batch in batches:
          character_embeds, primary_embeds = sess.run([self.character_lookup, self.position_lookup],
                                                      feed_dict={self.input_characters: batch['sentence'],
                                                                 self.input_position: batch['primary']})
          secondary_embeds = sess.run(self.position_lookup, feed_dict={self.input_position: batch['secondary']})
          input = sess.run(self.emebd_concat, feed_dict={self.character_embed_holder: character_embeds,
                                                         self.primary_embed_holder: primary_embeds,
                                                         self.secondary_embed_holder: secondary_embeds})
          # sess.run(self.train_model, feed_dict={self.input: input, self.input_relation: batch['label']})
          sess.run(self.train_cross_entropy_model, feed_dict={self.input: input, self.input_relation: batch['label']})
        self.saver.save(sess, self.output_folder + 'cnn_emr_model%d.ckpt' % i)

  def load_batches(self, path):
    with open(path, 'rb') as f:
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

  def predict(self, sentences, primary_indies, secondary_indices):
    with tf.Session() as sess:
      self.saver.restore(sess, self.output_folder + 'cnn_emr_model3.ckpt')
      character_embeds, primary_embeds = sess.run([self.character_lookup, self.position_lookup],
                                                  feed_dict={self.input_characters: sentences,
                                                             self.input_position: primary_indies})
      secondary_embeds = sess.run(self.position_lookup, feed_dict={self.input_position: secondary_indices})
      input = sess.run(self.emebd_concat, feed_dict={self.character_embed_holder: character_embeds,
                                                     self.primary_embed_holder: primary_embeds,
                                                     self.secondary_embed_holder: secondary_embeds})
      output = sess.run(self.output, feed_dict={self.input: input})
      return np.argmax(output, 1)

  def evaluate(self, model_file):
    with tf.Session() as sess:
      self.saver.restore(sess, self.output_folder + model_file)
      items = self.load_batches(self.test_batch_path)
      corr_count = [0] * self.relation_count
      prec_count = [0] * self.relation_count
      recall_count = [0] * self.relation_count

      for item in items:
        character_embeds, primary_embeds = sess.run([self.character_lookup, self.position_lookup],
                                                    feed_dict={self.input_characters: item['sentence'],
                                                               self.input_position: item['primary']})
        secondary_embeds = sess.run(self.position_lookup, feed_dict={self.input_position: item['secondary']})
        input = sess.run(self.emebd_concat, feed_dict={self.character_embed_holder: character_embeds,
                                                       self.primary_embed_holder: primary_embeds,
                                                       self.secondary_embed_holder: secondary_embeds})
        output = np.squeeze(sess.run(self.output, feed_dict={self.input: input}))
        target = np.nonzero(item['label'])[0][0]
        current = np.argmax(output)
        print(target, current)
        if target == current:
          corr_count[target] += 1
        prec_count[current] += 1
        recall_count[target] += 1

    prec = sum([c / p for c, p in zip(corr_count, prec_count)]) / self.relation_count
    recall = sum([c / r for c, r in zip(corr_count, recall_count)]) / self.relation_count
    print('precision:', prec)
    print('recall:', recall)


if __name__ == '__main__':
  reCNN = RECNN()
  reCNN.train()
