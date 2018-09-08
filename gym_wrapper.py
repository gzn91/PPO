import numpy as np
import cv2
from gym import spaces
from multiprocessing import Process, Pipe


def normalize(x, stats=None, eps=1e-8):
    if stats is None:
        return (x - x.mean()) / (x.std() + eps)
    else:
        return (x - stats.mean) / stats.std


# fn for multiprocessing, running multiple agents in parallel
def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'render':
            remote.send(env.render())
        else:
            raise NotImplementedError


# resize observation to grayscale 84x84
def preprocess_obs(obs):
    if np.ndim(obs) == 4:
        obs = obs[0]
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return obs[:, :, None]  # Shape (84, 84, 1)


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


# Stack multiple frames deepmind style
class StackObs(object):

    def __init__(self, env):
        self.env = env
        self.nstacks = 4
        self.ob_shape = (84, 84, self.nstacks)
        self.obs_max = np.zeros((self.nstacks // 2, *self.ob_shape[:-1], 1), dtype=np.uint8)
        self.obs = np.zeros(self.ob_shape, dtype=np.uint8)

        self.ret = 0
        self.lives = 0
        self.rewards = []

        self.gamma = .99
        self.epsilon = 1e-8
        self.clipob = 10.
        self.cliprew = 10.

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=255, shape=self.ob_shape, dtype=np.uint8)

    @property
    def action_space(self):
        return self.env.action_space

    def _step(self, actions):
        ob, rew, done, info = self.env.step(actions)
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return ob, rew, done, info

    # step environment, the agent has only one life even if environment has more.
    def step(self, actions):

        nstacks = self.nstacks
        rews = 0
        for i in range(nstacks):
            ob, rew, done, info = self._step(actions)

            rews += rew

            if i >= self.nstacks // 2:
                self.obs_max[i % (self.nstacks // 2)] = preprocess_obs(ob)

            if done:
                break

        self.rewards.append(rews)

        if done:
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            info = {"r": round(eprew, 6), "l": round(eplen)}
            self.reset()
        else:
            # roll/shift last channel axis by -1 so the oldest observations are removed first
            obs = self.obs_max.max(axis=0)
            self.obs = np.roll(self.obs, -1, -1)
            self.obs[..., -1:] = obs
            info = None

        return self.obs, rews, done, info

    def reset(self):
        self.rewards = []
        ob = preprocess_obs(self.env.reset())
        for i in range(self.nstacks):
            self.obs[..., i:] = ob
        self.lives = self.env.unwrapped.ale.lives()
        return self.obs

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    @staticmethod
    def expand_dim(arr):
        return np.expand_dims(arr, axis=0)


# TEST Environment for recurrent policys
class ResetEarly(object):

    def __init__(self, env):
        self.env = env
        self.ob_shape = (84, 84, 1)
        self.obs = np.zeros(self.ob_shape, dtype=np.uint8)
        self.obs_max = np.zeros((2, *self.ob_shape[:-1], 1), dtype=np.uint8)

        self.ret = 0
        self.lives = 0
        self.rewards = []
        self.nskips = 4


    @property
    def observation_space(self):
        return spaces.Box(low=0, high=255, shape=self.ob_shape, dtype=np.uint8)

    @property
    def action_space(self):
        return self.env.action_space

    def _step(self, actions):
        ob, rew, done, info = self.env.step(actions)
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return ob, rew, done, info

    # step environment, the agent has only one life even if environment has more.
    def step(self, actions):

        nskips = self.nskips
        rews = 0
        for i in range(nskips):
            ob, rew, done, info = self._step(actions)

            rews += rew

            if i >= self.nskips // 2:
                self.obs_max[i % (self.nskips // 2)] = preprocess_obs(ob)

            if done:
                break

        self.rewards.append(rews)

        if done:
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            info = {"r": round(eprew, 6), "l": round(eplen)}
            self.reset()
        else:
            self.obs[:] = self.obs_max.max(axis=0)
            info = None

        return self.obs, rews, done, info

    def reset(self):
        self.rewards = []
        ob = preprocess_obs(self.env.reset())
        self.lives = self.env.unwrapped.ale.lives()
        return self.obs

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    @staticmethod
    def expand_dim(arr):
        return np.expand_dims(arr, axis=0)

class GymWrap(object):

    def __init__(self, env):
        self.env = env
        self.ob_shape = self.env.observation_space.shape
        self.rewards = []

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        # print(ob.dtype)
        self.rewards.append(rew)

        if done:
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            info = {"r": round(eprew, 6), "l": eplen}
            ob = self.reset()
        else:
            info = None

        return ob.copy(), rew, done, info

    def reset(self):
        self.rewards = []
        return self.env.reset()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    @staticmethod
    def expand_dim(arr):
        return np.expand_dims(arr, axis=0)


class Workers(object):

    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        from time import sleep
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        self.nenvs = nenvs

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def render(self):
        self.remotes[0].send(('render', None))
        return self.remotes[0].recv()

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


# class Workers(object):
#
#     def __init__(self, env_fns):
#         self.nenvs = len(env_fns)
#         self.envs = [fn() for fn in env_fns]
#         if isinstance(self.envs[0].action_space, spaces.Discrete):
#             print('Discrete..')
#             ob_dtype = np.uint8
#         else:
#             print('Continuous..')
#             ob_dtype = np.float64
#         self.obs = np.zeros((self.nenvs, *self.observation_space.shape), dtype=ob_dtype)
#         self.rews = np.zeros((self.nenvs,), dtype=np.float32)
#         self.dones = np.zeros((self.nenvs,), dtype=np.bool)
#         self.infos = [{} for _ in range(self.nenvs)]
#         self.actions = None
#
#     @property
#     def observation_space(self):
#         return self.envs[0].observation_space
#
#     @property
#     def action_space(self):
#         return self.envs[0].action_space
#
#     def step_async(self, actions):
#         self.actions = actions
#
#     def step_wait(self):
#         for idx, (action, env) in enumerate(zip(self.actions, self.envs)):
#             self.obs[idx], self.rews[idx], self.dones[idx], self.infos[idx] = env.step(action)
#
#         return self.obs.copy(), self.rews.copy(), self.dones.copy(), self.infos
#
#     def step(self, actions):
#         self.step_async(actions)
#         return self.step_wait()
#
#     def reset(self):
#         for idx, env in enumerate(self.envs):
#             self.obs[idx] = env.reset()
#         return self.obs.copy()
#
#     def render(self):
#         return self.envs[0].render()



if __name__ == '__main__':
    envs = GymWrapNStack('BreakoutNoFrameskip-v4', 5, 4, 0)
    print(len(envs))
