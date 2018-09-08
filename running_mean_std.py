import tensorflow as tf, numpy as np


# tf, np implementation of OpenAI baselines running_mean_std
class Stats:

    def __init__(self, sess, shape=()):
        self.sess = sess
        self.sum = tf.get_variable(
            dtype=tf.float32,
            shape=shape,
            initializer=tf.constant_initializer(0.0),
            name='runningsum', trainable=False)
        self.sqsum = tf.get_variable(
            dtype=tf.float32,
            shape=shape,
            initializer=tf.constant_initializer(1e-2),
            name='runningsqsum', trainable=False)
        self.count = tf.get_variable(
            dtype=tf.float32,
            shape=(),
            initializer=tf.constant_initializer(1e-2),
            name='count', trainable=False)

        self.mean = tf.div(self.sum, self.count)
        self.std = tf.clip_by_value(tf.sqrt(
            tf.squared_difference(
                tf.sqrt(tf.div(self.sqsum, self.count)), self.mean)), 1e-2, 10)

        self.sum_next = tf.placeholder(tf.float32, shape=shape)
        self.sqsum_next = tf.placeholder(tf.float32, shape=shape)
        self.count_next = tf.placeholder(tf.float32, shape=())
        self.update_op = [self.sum.assign_add(self.sum_next), self.sqsum.assign_add(self.sqsum_next),
                          self.count.assign_add(self.count_next)]

    def update(self, state):
        self.sess.run(self.update_op, feed_dict={self.sum_next: state.sum(axis=0),
                                                 self.sqsum_next:  np.square(state).sum(axis=0),
                                                 self.count_next: np.array(len(state), dtype='float32')})
