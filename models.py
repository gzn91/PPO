import tensorflow as tf, numpy as np
import layers


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std

class RoboschoolGRU(object):

    def __init__(self, obs_rms, ob_shape, ac_shape, normalize_observations=True,
                 observation_range=(-5., 5.), reuse=False, nlstm=64, **kwargs):
        self.sess = tf.get_default_session()
        nbatch, nenvs = kwargs.values()

        obs_ph = tf.placeholder(tf.float32, [nbatch, *ob_shape], 'obs')  # obs
        mask_ph = tf.placeholder(tf.float32, [nbatch], 'masks')  # mask (done t-1)

        """Seperate GRU_Layer for policy fn and value fn"""
        state_ph = tf.placeholder(tf.float32, [nenvs, nlstm * 2], 'states')  # states
        state_pi, state_vf = tf.split(state_ph, 2, axis=-1)

        if normalize_observations:
            x = tf.clip_by_value(normalize(obs_ph, obs_rms),
                                 observation_range[0], observation_range[1])
        else:
            x = obs_ph

        with tf.variable_scope('model', reuse=reuse):
            with tf.variable_scope('actor'):
                h1 = layers.fc(x, 64, 'fc1')
                h2 = layers.fc(h1, 64, 'fc2')

                xs = layers.batch_to_seq(h2, nenvs, nbatch // nenvs)
                ms = layers.batch_to_seq(mask_ph, 1, nbatch)

                h3 = layers.gru_(xs, ms, state_pi, 'gru', nh=nlstm, reuse=reuse)
                h3_pi = layers.seq_to_batch(h3)

                mean = layers.fc(h3_pi, ac_shape[-1], 'mean', activate=False)
                log_stddev = tf.get_variable(name='log_stdev', shape=ac_shape, dtype=tf.float32,
                                             initializer=tf.zeros_initializer())

            with tf.variable_scope('critic', reuse=reuse):
                h1 = layers.fc(x, 64, 'fc1')
                h2 = layers.fc(h1, 64, 'fc2')

                xs = layers.batch_to_seq(h2, nenvs, nbatch // nenvs)
                ms = layers.batch_to_seq(mask_ph, 1, nbatch)

                h3 = layers.gru_(xs, ms, state_vf, 'gru', nh=nlstm, reuse=reuse)
                h3_vf = layers.seq_to_batch(h3)

                vf = layers.fc(h3_vf, 1, 'vf', activate=False)[:,0]

        self.h3 = tf.concat([h3_pi, h3_vf], axis=-1)

        self.initial_state = np.zeros((nbatch, nlstm * 2), dtype=np.float32)
        """End"""

        stddev = tf.exp(log_stddev)

        def sample():
            return tf.random_normal(ac_shape, mean=mean, stddev=stddev)

        def entropy():
            return tf.reduce_sum(log_stddev + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

        def kl(other):
            return tf.reduce_sum(
                other.log_stddev - log_stddev + (tf.square(stddev) + tf.square(mean - other.mean)) / (
                        2.0 * tf.square(other.stddev)) - 0.5, axis=-1)

        def neglogp(x):
            return 0.5 * tf.reduce_sum(tf.square((x - mean) / stddev), axis=-1) \
                   + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
                   + tf.reduce_sum(log_stddev, axis=-1)

        self.vf0 = vf
        self.a0 = sample()
        self.neglogp0 = neglogp(self.a0)
        self.mean = mean
        self.log_stddev = log_stddev
        self.stddev = stddev
        self.entropy = entropy
        self.kl = kl
        self.neglogp = neglogp
        self.obs_ph = obs_ph
        self.mask_ph = mask_ph
        self.state_ph = state_ph

    def step(self, ob, state, mask):
        # print(ob.shape,state.shape,mask.shape)
        return self.sess.run([self.a0, self.vf0, self.h3, self.neglogp0], {self.obs_ph: ob, self.state_ph: state, self.mask_ph: mask})

    def get_value(self, ob, state, mask):
        return self.sess.run(self.vf0, {self.obs_ph: ob, self.state_ph: state, self.mask_ph: mask})


class AtariGRU(object):

    def __init__(self, ob_shape, ac_shape, reuse=False, nlstm=256, **kwargs):
        self.sess = tf.get_default_session()
        nbatch, nenvs = kwargs.values()

        obs_ph = tf.placeholder(tf.uint8, [nbatch, *ob_shape], 'obs')  # obs
        mask_ph = tf.placeholder(tf.float32, [nbatch], 'masks')  # mask (done t-1)

        """Share GRU-Layer policy fn and value fn"""
        state_ph = tf.placeholder(tf.float32, [nenvs, nlstm], 'states')  # states

        with tf.variable_scope('model', reuse=reuse):

            # cast observations to float
            if obs_ph.dtype != tf.float32:
                x = tf.cast(obs_ph, tf.float32) / 255.

            # 3 layer conv2d + 2 layer fc
            h = layers.conv2d_block(x)

            # split h and mask into sentential list
            xs = layers.batch_to_seq(h, nenvs, nbatch//nenvs)
            ms = layers.batch_to_seq(mask_ph, 1, nbatch)

            # gated recurrent unit layer
            with tf.variable_scope('gru'):
                h6 = layers.gru_(xs, ms, state_ph, 'gru', nh=nlstm, use_ln=False, reuse=reuse)

            # reshape output to batch-tensor again
            h6 = layers.seq_to_batch(h6)

            with tf.variable_scope('actor'):
                logits = layers.fc(h6, ac_shape[-1], 'logits', activate=False, gain=0.01)

            with tf.variable_scope('critic'):
                vf = layers.fc(h6, 1, 'vf', activate=False, gain=1.0)[:,0]

        self.h6 = h6
        self.initial_state = np.zeros((nbatch, nlstm), dtype=np.float32)

        def sample():
            u = tf.random_uniform(tf.shape(logits))
            return tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)

        def entropy():
            a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

        def kl(other):
            a0 = logits - tf.reduce_max(logits, axis=-1, keep_dims=True)
            a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keep_dims=True)
            ea0 = tf.exp(a0)
            ea1 = tf.exp(a1)
            z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
            z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

        def neglogp(x):
            one_hot_actions = tf.one_hot(x, logits.get_shape().as_list()[-1])
            return tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits,
                labels=one_hot_actions)

        self.a0 = sample()
        self.neglogp0 = neglogp(self.a0)
        self.vf0 = vf
        self.entropy = entropy
        self.kl = kl
        self.neglogp = neglogp
        self.obs_ph = obs_ph
        self.mask_ph = mask_ph
        self.state_ph = state_ph

    def step(self, ob, state, mask):
        return self.sess.run([self.a0, self.vf0, self.h6, self.neglogp0], {self.obs_ph: ob, self.state_ph: state, self.mask_ph: mask})

    def get_value(self, ob, state, mask):
        return self.sess.run(self.vf0, {self.obs_ph: ob, self.state_ph: state, self.mask_ph: mask})


class AtariLSTM(object):

    def __init__(self, ob_shape, ac_shape, reuse=False, nlstm=256, **kwargs):
        self.sess = tf.get_default_session()
        nbatch, nenvs = kwargs.values()

        obs_ph = tf.placeholder(tf.uint8, [nbatch, *ob_shape], 'obs')  # obs
        mask_ph = tf.placeholder(tf.float32, [nbatch], 'masks')  # mask (done t-1)

        """Share LSTM-Layer policy fn and value fn"""
        state_ph = tf.placeholder(tf.float32, [nenvs, nlstm*2], 'states')  # states

        with tf.variable_scope('model', reuse=reuse):

            # cast observations to float
            if obs_ph.dtype != tf.float32:
                x = tf.cast(obs_ph, tf.float32) / 255.

            # 3 layer conv2d + 2 layer fc
            h = layers.conv2d_block(x)

            # split h and mask into sentential list
            xs = layers.batch_to_seq(h, nenvs, nbatch//nenvs)
            ms = layers.batch_to_seq(mask_ph, 1, nbatch)

            # long short-term memory layer
            h6, snew = layers.lstm(xs, ms, state_ph, 'lstm', nh=nlstm, use_ln=False, reuse=reuse)

            # reshape output to batch-tensor again
            h6 = layers.seq_to_batch(h6)

            with tf.variable_scope('actor'):
                logits = layers.fc(h6, ac_shape[-1], 'logits', activate=False, gain=0.01)

            with tf.variable_scope('critic'):
                vf = layers.fc(h6, 1, 'vf', activate=False, gain=1.0)[:,0]

        self.snew = snew
        self.initial_state = np.zeros((nbatch, nlstm*2), dtype=np.float32)

        def sample():
            u = tf.random_uniform(tf.shape(logits))
            return tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)

        def entropy():
            a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

        def kl(other):
            a0 = logits - tf.reduce_max(logits, axis=-1, keep_dims=True)
            a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keep_dims=True)
            ea0 = tf.exp(a0)
            ea1 = tf.exp(a1)
            z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
            z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

        def neglogp(x):
            one_hot_actions = tf.one_hot(x, logits.get_shape().as_list()[-1])
            return tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits,
                labels=one_hot_actions)

        self.a0 = sample()
        self.neglogp0 = neglogp(self.a0)
        self.vf0 = vf
        self.entropy = entropy
        self.kl = kl
        self.neglogp = neglogp
        self.obs_ph = obs_ph
        self.mask_ph = mask_ph
        self.state_ph = state_ph

    def step(self, ob, state, mask):
        return self.sess.run([self.a0, self.vf0, self.snew, self.neglogp0], {self.obs_ph: ob, self.state_ph: state, self.mask_ph: mask})

    def get_value(self, ob, state, mask):
        return self.sess.run(self.vf0, {self.obs_ph: ob, self.state_ph: state, self.mask_ph: mask})


class AtariCNN(object):
    def __init__(self, ob_shape, ac_shape, reuse=False, **kwargs):
        self.sess = tf.get_default_session()
        nbatch, nenvs = kwargs.values()

        obs_ph = tf.placeholder(tf.uint8, [nbatch,*ob_shape], 'obs_ph')

        with tf.variable_scope('model', reuse=reuse):

            if obs_ph.dtype != tf.float32:
                x = tf.cast(obs_ph, tf.float32) / 255.

            with tf.variable_scope('cnn'):
                h = layers.conv2d_block(x)

            with tf.variable_scope('actor'):
                logits = layers.fc(h, ac_shape[-1], 'logits', activate=False, gain=0.01)

            with tf.variable_scope('critic'):
                vf = layers.fc(h, 1, 'vf', activate=False, gain=1.0)[:,0]

        def sample():
            u = tf.random_uniform(tf.shape(logits))
            return tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)

        def entropy():
            a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

        def kl(other):
            a0 = logits - tf.reduce_max(logits, axis=-1, keep_dims=True)
            a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keep_dims=True)
            ea0 = tf.exp(a0)
            ea1 = tf.exp(a1)
            z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
            z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

        def neglogp(x):
            one_hot_actions = tf.one_hot(x, logits.get_shape().as_list()[-1])
            return tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits,
                labels=one_hot_actions)

        self.a0 = sample()
        self.neglogp0 = neglogp(self.a0)
        self.vf0 = vf
        self.entropy = entropy
        self.kl = kl
        self.neglogp = neglogp
        self.obs_ph = obs_ph
        self.initial_state = None

    def step(self, ob, *_args):
        action, value, neglogp = self.sess.run([self.a0, self.vf0, self.neglogp0], feed_dict={self.obs_ph: ob})
        return action, value, self.initial_state, neglogp

    def get_value(self, ob, *_args):
        return self.sess.run(self.vf0, feed_dict={self.obs_ph: ob})


class RoboschoolMLP(object):
    def __init__(self, obs_rms, ob_shape, ac_shape, normalize_observations=True, observation_range=(-5., 5.), reuse=False, **kwargs):
        self.sess = tf.get_default_session()
        nbatch, nenvs = kwargs.values()
        obs_ph = tf.placeholder(tf.float32, [nbatch,*ob_shape], 'obs_ph')

        with tf.variable_scope('model', reuse=reuse):
            if normalize_observations:
                x = tf.clip_by_value(normalize(obs_ph, obs_rms),
                                       observation_range[0], observation_range[1])
            else:
                x = obs_ph
            with tf.variable_scope('actor'):
                with tf.variable_scope('hidden'):
                    h1 = layers.fc(x, 64, 'fc1')
                    h2 = layers.fc(h1, 64, 'fc2')
                mean = layers.fc(h2, ac_shape[-1], 'mean', activate=False)
                log_stddev = tf.get_variable(name='log_stdev', shape=ac_shape, dtype=tf.float32,
                                             initializer=tf.zeros_initializer())
            with tf.variable_scope('critic'):
                with tf.variable_scope('hidden'):
                    h1 = layers.fc(x, 64, 'fc1')
                    h2 = layers.fc(h1, 64, 'fc2')
                vf = layers.fc(h2, 1, 'vf', activate=False, gain=1.0)[:,0]

        stddev = tf.exp(log_stddev)

        def sample():
            return tf.random_normal(ac_shape, mean=mean, stddev=stddev)

        def entropy():
            return tf.reduce_sum(log_stddev + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

        def kl(other):
            return tf.reduce_sum(
                other.log_stddev - log_stddev + (tf.square(stddev) + tf.square(mean - other.mean)) / (
                        2.0 * tf.square(other.stddev)) - 0.5, axis=-1)

        def neglogp(x):
            return 0.5 * tf.reduce_sum(tf.square((x - mean) / stddev), axis=-1) \
                   + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
                   + tf.reduce_sum(log_stddev, axis=-1)

        self.vf0 = vf
        self.a0 = sample()
        self.neglogp0 = neglogp(self.a0)
        self.mean = mean
        self.log_stddev = log_stddev
        self.stddev = stddev
        self.entropy = entropy
        self.kl = kl
        self.neglogp = neglogp
        self.obs_ph = obs_ph
        self.initial_state = None

    def step(self, ob, *_args):
        action, values, neglogp = self.sess.run([self.a0, self.vf0, self.neglogp0], feed_dict={self.obs_ph: ob})
        return action, values, self.initial_state, neglogp

    def get_value(self, ob, *_args):
        return self.sess.run(self.vf0, feed_dict={self.obs_ph: ob})