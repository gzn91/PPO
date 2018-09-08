import tensorflow as tf
import numpy as np
from running_mean_std import Stats
import models
from gym import spaces
import time
from collections import deque
from datetime import datetime
from time import sleep

flags = tf.app.flags
FLAGS = flags.FLAGS


class PPO(object):

    def __init__(self, env_fn, learning_rate=3e-4, clip_range=0.2, max_steps=1e6, nsteps=2056, mb_size=256,
                 opteps=4, vf_coef=0.5, ent_coef=0.01, gae=0.95, gamma=0.99, normalize_observations=False,
                 observation_range=(-10., 10.)):

        self.sess = tf.get_default_session()
        self.env = env_fn
        self.nenvs = self.env.nenvs
        self.gae = gae
        self.gamma = gamma
        self.global_step = tf.Variable(0)
        self.step = self.global_step.assign_add(1)
        ob_shape = self.env.observation_space.shape

        self.max_steps = int(max_steps)
        self.nsteps = nsteps
        self.nbatch = nsteps * self.nenvs
        print('nbatch:', self.nbatch)
        self.mb_size = mb_size
        self.opteps = opteps

        train_kwargs = {'nbatch': self.mb_size, 'nenvs': self.nenvs}
        act_kwargs = {'nbatch': self.nenvs, 'nenvs': self.nenvs}

        self.obs_rms = None
        self.normalize_observations = normalize_observations
        if normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = Stats(self.sess, shape=ob_shape)

        # Create correct model depending on env
        # Need 2 models (act and train) since gru-layer cant handle dynamic shapes
        if isinstance(self.env.action_space, spaces.Discrete):
            ac_shape = (self.env.action_space.n,)
            dtype_obs = np.uint8
            self.dtype_act = np.int32
            self.act_shape = ()

            with tf.variable_scope('model'):

                self.act_model = models.AtariCNN(ob_shape, ac_shape, **act_kwargs)
                train_model = models.AtariCNN(ob_shape, ac_shape, **train_kwargs, reuse=True)
                self.ac_ph = tf.placeholder(tf.int32, [None], 'ac_ph')

        else:
            ac_shape = self.env.action_space.shape
            dtype_obs = np.float32
            self.dtype_act = np.float32
            self.act_shape = ac_shape

            with tf.variable_scope('model'):
                self.act_model = models.RoboschoolMLP(self.obs_rms, ob_shape, ac_shape, normalize_observations,
                                                      observation_range, **act_kwargs)
                train_model = models.RoboschoolMLP(self.obs_rms, ob_shape, ac_shape, normalize_observations,
                                                   observation_range, **train_kwargs, reuse=True)
                self.ac_ph = tf.placeholder(tf.float32, [None, *ac_shape], 'ac_ph')

        # placeholders
        self.returns_ph = tf.placeholder(tf.float32, [None], 'returns_ph')
        self.adv_ph = tf.placeholder(tf.float32, [None], 'adv_ph')
        self.oldneglogp_ph = tf.placeholder(tf.float32, [None], 'oldneglogp_ph')
        self.oldvf_ph = tf.placeholder(tf.float32, [None], 'oldvf_ph')
        self.lr_ph = tf.placeholder(tf.float32, [], 'lr_ph')
        self.clip_ph = tf.placeholder(tf.float32, [], 'clip_ph')
        self.reward_ph = tf.placeholder(tf.float32, [], 'clip_ph')

        # create reward summary
        tf.summary.scalar('mean_reward', self.reward_ph)

        self.model = train_model

        # surrogate objective
        neglogp = self.model.neglogp(self.ac_ph)
        ratio = tf.exp(self.oldneglogp_ph - neglogp)
        clipped_ratio = tf.clip_by_value(ratio, 1 - clip_range, 1 + clip_range)
        surrogate_obj = tf.reduce_mean(tf.minimum(self.adv_ph * ratio, self.adv_ph * clipped_ratio))
        tf.summary.scalar('surrogate', surrogate_obj)

        # value function loss
        vf = self.model.vf0
        clipped_vf = self.oldvf_ph + tf.clip_by_value(vf - self.oldvf_ph, -clip_range, clip_range)
        vf_loss = .5 * tf.reduce_mean(
            tf.minimum(tf.square(vf - self.returns_ph), tf.square(clipped_vf - self.returns_ph)))
        tf.summary.scalar('vf-loss', vf_loss)

        # entropy bonus
        entropy = tf.reduce_mean(self.model.entropy())
        tf.summary.scalar('entropy', entropy)

        # kl
        kl = .5 * tf.reduce_mean(tf.square(neglogp - self.oldneglogp_ph))

        # total loss
        self.loss = -(surrogate_obj - vf_coef * vf_loss + ent_coef * entropy)

        # obj info
        self.infos = dict()
        info = [surrogate_obj, vf_loss, entropy, kl]
        info_names = ['surrogate', 'vf_loss', 'entropy', 'kl']
        for name, obj in zip(info_names, info):
            self.infos[name] = obj

        params = tf.trainable_variables(scope='model')

        # compute gradient
        grads = tf.gradients(self.loss, params, name='grads')
        grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
        grads_and_vars = list(zip(grads, params))

        # optimize
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_ph, epsilon=1e-5)
        self.optimize = optimizer.apply_gradients(grads_and_vars)
        # self.optimize = optimizer.minimize(self.loss)

        self.learning_rate = learning_rate
        self.clip_range = clip_range

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        self.summary = tf.Summary()
        self.merge_op = tf.summary.merge_all()
        print('Logging to:', './logs/'+str(datetime.now()))
        self.writer = tf.summary.FileWriter('./logs/'+str(datetime.now()))

        # buffers
        self.obs = np.zeros((self.nenvs, *ob_shape), dtype=dtype_obs)
        self.states = self.act_model.initial_state
        self.dones = np.asarray([False] * self.env.nenvs).reshape((self.env.nenvs,))

    def train(self, *slices):
        obs, advantages, values, actions, rewards, dones, neglogps, lr, cr, eprew = slices
        returns = advantages + values
        norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss = 0
        lossinfos = []

        batch_inds = np.arange(self.nbatch)

        # if model in not a RNN
        if self.states is None:
            for ep in range(self.opteps):
                np.random.shuffle(batch_inds)
                for start in range(0, self.nbatch, self.mb_size):
                    end = start + self.mb_size
                    inds = batch_inds[start:end]
                    fd_map = {self.model.obs_ph: obs[inds], self.reward_ph: eprew, self.ac_ph: actions[inds], self.adv_ph: norm_advantages[inds],
                              self.oldvf_ph: values[inds], self.oldneglogp_ph: neglogps[inds],
                              self.returns_ph: returns[inds], self.lr_ph: lr, self.clip_ph: cr}

                    _loss, _, summary, info = self.sess.run([self.loss, self.optimize, self.merge_op, self.infos],
                                                                 feed_dict=fd_map)
                    lossinfos.append(info)
                    loss += _loss

        # if model is a RNN we pick random samples from same env
        else:
            envsperbatch = self.mb_size // self.nsteps
            envinds = np.arange(self.nenvs)
            batch_inds = batch_inds.reshape((self.nenvs, self.nsteps))
            for ep in range(self.opteps):
                np.random.shuffle(envinds)
                for start in range(0, self.nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    inds = batch_inds[mbenvinds].ravel()

                    fd_map = {self.model.obs_ph: obs[inds], self.reward_ph: eprew, self.ac_ph: actions[inds], self.adv_ph: norm_advantages[inds],
                              self.oldvf_ph: values[inds], self.oldneglogp_ph: neglogps[inds],
                              self.returns_ph: returns[inds], self.lr_ph: lr, self.clip_ph: cr,
                              self.model.state_ph: self.states, self.model.mask_ph: dones[inds]}

                    _loss, _, summary, info = self.sess.run([self.loss, self.optimize, self.merge_op, self.infos], feed_dict=fd_map)
                    lossinfos.append(info)
                    loss += _loss

        """
        ** This code can run recurrent network when minibatch is less than nsteps, however it takes a lot of
        time for the first sess.run() **
        
        # if model is a RNN we pick random samples from same env
        else:
            # This is the case where mb_size < nsteps, often case in continuous environments
            envsperbatch = self.mb_size // self.nsteps
            envinds = np.arange(self.nenvs)
            batch_inds = batch_inds.reshape((self.nenvs, self.nsteps))
            for ep in range(self.opteps):
                if envsperbatch == 0:
                    for env_inds in batch_inds:
                        np.random.shuffle(env_inds)
                        for start in range(0, self.nsteps, self.mb_size):
                            end = start + self.mb_size
                            inds = env_inds[start:end]
                            fd_map = {self.model.obs_ph: obs[inds], self.ac_ph: actions[inds],
                                      self.adv_ph: norm_advantages[inds],
                                      self.oldvf_ph: values[inds], self.oldneglogp_ph: neglogps[inds],
                                      self.returns_ph: returns[inds], self.lr_ph: lr, self.clip_ph: cr,
                                      self.model.state_ph: self.states, self.model.mask_ph: dones[inds]}

                            _loss, _ = self.sess.run([self.loss, self.optimize], feed_dict=fd_map)
                            loss += _loss
                # This is the simple case where mb_size > nsteps such as in Atari, this enables us to
                # sample multiple environments in each minibatch.
                else:
                    assert self.mb_size % self.nsteps == 0
                    np.random.shuffle(envinds)
                    for start in range(0, self.nenvs, envsperbatch):
                        end = start + envsperbatch
                        mbenvinds = envinds[start:end]
                        inds = batch_inds[mbenvinds].ravel()

                        fd_map = {self.model.obs_ph: obs[inds], self.ac_ph: actions[inds], self.adv_ph: norm_advantages[inds],
                                  self.oldvf_ph: values[inds], self.oldneglogp_ph: neglogps[inds],
                                  self.returns_ph: returns[inds], self.lr_ph: lr, self.clip_ph: cr,
                                  self.model.state_ph: self.states, self.model.mask_ph: dones[inds]}

                        _loss, _ = self.sess.run([self.loss, self.optimize], feed_dict=fd_map)
                        loss += _loss
            """

        loss /= (self.opteps * self.nbatch) / self.mb_size
        infos = self.get_infos(lossinfos)
        return loss, summary, infos

    def run(self):
        if FLAGS.restore:
            self.restore_model()
        ep = self.sess.run(self.global_step)

        # decreasing learning rate and clip range
        lr = lambda x: (1 - x) * self.learning_rate
        cr = lambda x: (1 - x) * self.clip_range
        _step = 0.
        max_eps = self.max_steps // self.nbatch
        self.obs[:] = self.env.reset()

        epinfo_buf = deque(maxlen=100)

        while ep <= max_eps:

            time_start = time.time()
            frac = ep / max_eps
            lrnow = lr(frac)
            crnow = cr(frac)
            epinfos = []

            obs = np.zeros((self.nsteps, *self.obs.shape), dtype=self.obs.dtype)
            rewards = np.zeros((self.nsteps, self.nenvs), dtype=np.float32)
            actions = np.zeros((self.nsteps, self.nenvs, *self.act_shape), dtype=self.dtype_act)
            dones = np.zeros((self.nsteps, self.nenvs), dtype=np.bool)
            neglogps = np.zeros((self.nsteps, self.nenvs), dtype=np.float32)
            values = np.zeros((self.nsteps, self.nenvs), dtype=np.float32)

            for step in range(self.nsteps):
                if FLAGS.render:
                    self.env.render()
                    sleep(0.002)
                # print(step)

                obs[step] = self.obs

                actions[step], values[step], self.states, neglogps[step] = self.act_model.step(self.obs, self.states,
                                                                                               self.dones)
                self.obs[:], rewards[step], dones[step], infos = self.env.step(actions[step])
                self.dones = dones[step]

                [epinfos.append(info) for info in infos if info]

                if self.normalize_observations:
                    self.obs_rms.update(self.obs)

            epinfo_buf.extend(epinfos)

            eprew = self.safemean([epinfo['r'] for epinfo in epinfo_buf])
            eplen = self.safemean([epinfo['l'] for epinfo in epinfo_buf])

            if FLAGS.train:
                advantages = self.get_advantages(values, rewards, dones)
                slices = (
                    *map(self.flat_arr, (obs, advantages, values, actions, rewards, dones, neglogps)), lrnow, crnow, eprew)
                loss, summary, infos = self.train(*slices)
                time_end = time.time()
                fps = int(self.nenvs * self.nsteps * self.opteps // (time_end - time_start))
                print('*'*8, f'Episode: {ep}', '*'*8)
                for k, v in infos.items():
                    print(f'\t{k}: {v:.4f}')
                print(f'\tmean_reward: {eprew:.2f}')
                print(f'\tmean_length: {eplen:.2f}')
                print(f'\tfps: {fps}\n')

                if ep % 10 == 0:
                    self.writer.add_summary(summary,global_step=ep)
                    self.writer.add_summary(self.summary,global_step=ep)

            else:
                print(
                    f'|| avg reward {eprew:.4f} || avg eplength: {eplen:.4f} ||')
            ep = self.sess.run(self.step)

            _step = ep * self.nsteps
            if ep % 200 == 0:
                self.save_model()

    def flat_arr(self, arr):
        arr = np.reshape(arr, [arr.shape[0] * arr.shape[1], *arr.shape[2:]])
        return arr

    def safemean(self, xs):
        return np.nan if len(xs) == 0 else np.mean(xs)

    def get_advantages(self, values, rewards, dones):
        mask = 1 - dones
        lastvalues = self.act_model.get_value(self.obs, self.states, self.dones)
        prevgae = 0
        advantages = np.zeros_like(values)
        for t in reversed(range(values.shape[0])):
            delta = rewards[t, :] + self.gamma * lastvalues * mask[t, :] - values[t, :]
            advantages[t, :] = prevgae = delta + self.gae * self.gamma * mask[t, :] * prevgae
            lastvalues = values[t, :]
        return advantages

    def get_infos(self, infos):
        d = dict()
        for n, info in enumerate(infos):
            for k, v in info.items():
                if k in d:
                    d[k] += v/len(infos)
                    continue
                d[k] = v/len(infos)

        return d

    def restore_model(self):
        self.saver.restore(self.sess, '{}'.format(tf.train.latest_checkpoint('./saved_model/')))

    def save_model(self):
        self.saver.save(self.sess, './saved_model/model', global_step=self.sess.run(self.global_step))

