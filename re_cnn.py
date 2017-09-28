# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import pickle


class RECNN():
  def __init__(self, relation_count=2, window_size=(3,), batch_size=50, batch_length=85,train=True):
    tf.reset_default_graph()
    self.dtype = tf.float32
    self.window_size = window_size
    self.filter_size = 150
    self.relation_count = relation_count
    self.batch_length = batch_length
    self.batch_size = batch_size
    self.learning_rate = 0.01
    self.dropout_rate = 0.5
    self.lam = 0.0001
    self.character_embed_size = 300
    self.position_embed_size = 50
    self.dict_path = 'corpus/emr_words_dict.utf8'
    self.dictionary = self.read_dictionary()
    self.words_size = len(self.dictionary)
    self.is_train = train
    if relation_count == 2:
      self.batch_path = 'corpus/emr_all_relation_batches.rel'
      self.output_folder = 'tmp/re_two/'
      self.test_batch_path = 'corpus/emr_test_all_relations.rel'
    elif relation_count == 29:
      self.batch_path = 'corpus/emr_relation_batches.rel'
      self.output_folder = 'tmp/re_multi/'
      self.test_batch_path = 'corpus/emr_test_relations.rel'
    else:
      raise Exception('relation count error')

    self.concat_embed_size = self.character_embed_size + 2 * self.position_embed_size
    self.input_characters = tf.placeholder(tf.int32, [None, self.batch_length])
    self.input_position = tf.placeholder(tf.int32, [None, self.batch_length])
    self.input = tf.placeholder(self.dtype, [None, self.batch_length, self.concat_embed_size, 1])
    self.input_relation = tf.placeholder(self.dtype, [None, self.relation_count])
    self.position_embedding = self.weight_variable([2 * self.batch_length, self.position_embed_size])
    self.character_embedding = self.weight_variable([self.words_size, self.character_embed_size])
    self.conv_kernel = self.get_conv_kernel()
    self.bias = [self.weight_variable([self.filter_size])] * len(self.window_size)
    self.full_connected_weight = self.weight_variable([self.filter_size*len(self.window_size), self.relation_count])
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
    if train:
      self.hidden_layer = tf.layers.dropout(self.get_hidden(), self.dropout_rate)
    else:
      self.hidden_layer = tf.expand_dims(tf.layers.dropout(self.get_hidden(), self.dropout_rate),0)
    self.output_no_softmax = tf.matmul(self.hidden_layer, self.full_connected_weight) + self.full_connected_bias
    self.output = tf.nn.softmax(tf.matmul(self.hidden_layer, self.full_connected_weight) + self.full_connected_bias)
    self.params = [self.position_embedding, self.character_embedding, self.full_connected_weight,
                   self.full_connected_bias] + self.conv_kernel + self.bias
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

  def get_conv_kernel(self):
    conv_kernel = []
    for w in self.window_size:
      conv_kernel.append(self.weight_variable([w, self.concat_embed_size, 1, self.filter_size]))
    return conv_kernel

  def get_max_pooling(self, x):
    max_pooling = []
    for w in self.window_size:
      max_pooling.append(self.max_pooling(x, w))
    return max_pooling

  def get_hidden(self):
    h = None
    for w, conv, bias in zip(self.window_size, self.conv_kernel, self.bias):
      if h is None:
        h = tf.squeeze(self.max_pooling(tf.nn.relu(self.conv(conv) + bias), w))
      else:
        hh = tf.squeeze(self.max_pooling(tf.nn.relu(self.conv(conv) + bias), w))
        if self.is_train:
          h = tf.concat([h, hh], 1)
        else:
          h = tf.concat([h,hh], 0)
    return h

  def conv(self, conv_kernel):
    return tf.nn.conv2d(self.input, conv_kernel, strides=[1, 1, 1, 1], padding='VALID')

  def max_pooling(self, x, window_size):
    return tf.nn.max_pool(x, ksize=[1, self.batch_length - window_size + 1, 1, 1],
                          strides=[1, 1, 1, 1], padding='VALID')

  def train(self):
    batches = self.load_batches(self.batch_path)
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      sess.graph.finalize()
      epoches = 100
      for i in range(1, epoches + 1):
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
        if i % 50 == 0:
          model_name = 'cnn_emr_model{0}_{1}.ckpt'.format(i, '_'.join(map(str, self.window_size)))
          self.saver.save(sess, self.output_folder + model_name)

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
    #tf.reset_default_graph()
    with tf.Session() as sess:
      #tf.global_variables_initializer().run()

      self.saver.restore(sess=sess, save_path=self.output_folder + model_file)
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
        # print(input)
        output = np.squeeze(sess.run(self.output, feed_dict={self.input: input}))
        target = np.argmax(item['label'])
        current = np.argmax(output)
        if target == current:
          corr_count[target] += 1
        prec_count[current] += 1
        recall_count[target] += 1

    precs = [c / p for c, p in zip(corr_count, prec_count) if p != 0 and c != 0]
    recalls = [c / r for c, r in zip(corr_count, recall_count) if r!= 0 and c != 0]
    print(corr_count)
    print(recall_count)
    print(corr_count)
    print(precs)
    print(recalls)
    prec = sum(precs) / len(precs)
    recall = sum(recalls) / len(recalls)
    f1 = 2*prec*recall/(prec+recall)
    print('precision:', prec)
    print('recall:', recall)
    print('f1',f1)

def train_two():
  re_2 = RECNN(window_size=(2,))
  re_2.train()
  re_3 = RECNN(window_size=(3,))
  re_3.train()
  re_4 = RECNN(window_size=(4,))
  re_4.train()
  re_2_3 = RECNN(window_size=(2, 3))
  re_2_3.train()
  re_3_4 = RECNN(window_size=(3, 4))
  re_3_4.train()
  re_2_3_4 = RECNN(window_size=(2, 3, 4))
  re_2_3_4.train()

def train_multi():
  re_2 = RECNN(window_size=(2,),relation_count=29)
  re_2.train()
  re_3 = RECNN(window_size=(3,),relation_count=29)
  re_3.train()
  re_4 = RECNN(window_size=(4,),relation_count=29)
  re_4.train()
  re_2_3 = RECNN(window_size=(2, 3),relation_count=29)
  re_2_3.train()
  re_3_4 = RECNN(window_size=(3, 4),relation_count=29)
  re_3_4.train()
  re_2_3_4 = RECNN(window_size=(2, 3, 4),relation_count=29)
  re_2_3_4.train()

if __name__ == '__main__':
  train_two()
  train_multi()
