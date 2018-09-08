import gym, atari_py, roboschool
import gym_wrapper
from ppo import PPO
import tensorflow as tf, numpy as np
import random


def main(_):

    tf.Session().__enter__()
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    def make_env(seed):
        def _make_env():
            env = gym.make(FLAGS.env)
            env.seed(seed)
            env.allow_early_resets = True
            env = gym_wrapper.StackObs(env)

            return env

        return _make_env

    try:
        env = gym_wrapper.Workers([make_env(_) for _ in [random.randint(0, 1000)] * FLAGS.nenvs])

        ppo = PPO(env, nsteps=FLAGS.nsteps, learning_rate=FLAGS.lr, clip_range=FLAGS.cr,
                  max_steps=FLAGS.max_steps, mb_size=FLAGS.mb_size, opteps=FLAGS.opteps,
                  gae=FLAGS.gae, gamma=FLAGS.gamma, vf_coef=FLAGS.vf_coef, ent_coef=FLAGS.ent_coef,
                  normalize_observations=FLAGS.normalize_obs)
        ppo.run()
        env.close()

    except KeyboardInterrupt:
        env.close()


if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    # Environment flags
    flags.DEFINE_bool('train', False, 'If model should be trained.')
    flags.DEFINE_bool('restore', True, 'If restore previous model.')
    flags.DEFINE_bool('render', True, 'If render the environment.')
    flags.DEFINE_string('env', 'BreakoutNoFrameskip-v4', 'Environment that should be used.')
    flags.DEFINE_integer('seed', 1337, 'Random seed to be used.')
    flags.DEFINE_integer('nenvs', 8, 'Number of parallel environments.')
    flags.DEFINE_bool('normalize_obs', False, 'If observations should be normalized.')

    # Training flags
    flags.DEFINE_integer('max_steps', int(1e7), 'Total steps to be performed.')
    flags.DEFINE_integer('opteps', 4, 'Total optimization epochs per episode to be performed.')
    flags.DEFINE_integer('nsteps', 128, 'Number of steps in each episode.')
    flags.DEFINE_integer('mb_size', 256, 'Number of samples in each minibatch.')
    flags.DEFINE_float('lr', 3e-4, 'Learning rate.')
    flags.DEFINE_float('cr', 0.1, 'Clip range.')
    flags.DEFINE_float('gae', 0.95, 'Generalized Advantage Estimation coefficient.')
    flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
    flags.DEFINE_float('vf_coef', 0.5, 'Value loss coefficient.')
    flags.DEFINE_float('ent_coef', 0.01, 'Entropy bonus coefficient.')

    tf.app.run()