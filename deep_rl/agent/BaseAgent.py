#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import numpy as np
from ..utils import *
import torch.multiprocessing as mp
from collections import deque
from skimage.io import imsave
from tqdm import tqdm
from copy import deepcopy

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(game=config.game_file, tag=config.logtxtname, log_level=config.log_level)
        self.task_ind = 0

        self.adv_attack = config.adv_attack
        self.epsilon = config.epsilon
        self.PGD_iter = config.PGD
        if self.PGD_iter == 3: # default
            self.step_size = self.epsilon * 1.0 / 4
        else: # 10: for SA-DQN attack
            self.epsilon = 1 / 255.0
            self.step_size = self.epsilon * 1.0 / self.PGD_iter

    def close(self):
        close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)


    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state):
        raise NotImplementedError

    def eval_step_adv(self, state):
        raise NotImplementedError

    def eval_step_normality(self, state):
        raise NotImplementedError

    def eval_normality(self):
        # (1) fix two states
        env0 = self.config.eval_env
        state = env0.reset()
        state0 = deepcopy(state)
        while True:
            action = self.eval_step(state)  # for different agents
            state, reward, done, info = env0.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        state1 = deepcopy(state)
        self.config.state_normalizer.set_read_only()
        state0 = self.config.state_normalizer(state0)
        state1 = self.config.state_normalizer(state1)
        self.config.state_normalizer.unset_read_only()
        # (2) linear interpolation
        t = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        r = {}
        for i in t:
            r[i] = []
        for i in range(len(t)):
            print(i, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            i_t = t[i]
            for ep in tqdm(range(self.config.eval_episodes)):  # loop for many episodes
                env = self.config.eval_env
                # state = env.reset()
                state_interpolate = (1-i_t) * state0 + i_t * state1 # numpy
                action = self.eval_step_normality(state_interpolate)
                state_interpolate, _, _, _ = env.step(action)

                while True:
                    action = self.eval_step(state_interpolate)
                    state_interpolate, reward, done, info = env.step(action)
                    ret = info[0]['episodic_return']
                    if ret is not None:
                        break
                r[i_t].append(np.sum(ret))

        print(r)

    def eval_episode(self): # evaluation within each episode
        env = self.config.eval_env
        state = env.reset()
        while True:
            if self.adv_attack:
                action = self.eval_step_adv(state) # for different agents
            else:
                action = self.eval_step(state) # for different agents
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        # loop to the end to get the return
        return ret

    def eval_episodes(self):
        episodic_returns = []
        for ep in tqdm(range(self.config.eval_episodes)): # loop for many episodes
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
            self.total_steps, np.mean(episodic_returns), np.std(episodic_returns) / np.sqrt(len(episodic_returns)) # std for mean
        ))
        self.logger.add_scalar('episodic_return_test', np.mean(episodic_returns), self.total_steps)
        return {
            'episodic_return_test': np.mean(episodic_returns),
            'episodic_return_test_std': np.std(episodic_returns) / np.sqrt(len(episodic_returns)),
        }



    def record_online_return(self, info, offset=0):
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, ret))
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def switch_task(self):
        config = self.config
        if not config.tasks:
            return
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)

    def record_episode(self, dir, env):
        mkdir(dir)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, dir, steps)
            action = self.record_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            steps += 1
            if ret is not None:
                break

    def record_step(self, state):
        raise NotImplementedError

    # For DMControl
    def record_obs(self, env, dir, steps):
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave('%s/%04d.png' % (dir, steps), obs)


class BaseActor(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config):
        mp.Process.__init__(self)
        self.config = config
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self._state = None
        self._task = None
        self._network = None
        self._total_steps = 0
        self.__cache_len = 2

        if not config.async_actor:
            self.start = lambda: None # multiprocess cannot pickle lambda
            self.step = self._sample
            self.close = lambda: None
            self._set_up()
            self._task = config.task_fn()

    def _sample(self):
        transitions = []
        for _ in range(self.config.sgd_update_frequency):
            transition = self._transition()
            if transition is not None:
                transitions.append(transition)
        return transitions

    def run(self):
        self._set_up()
        config = self.config
        self._task = config.task_fn()

        cache = deque([], maxlen=2)
        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.STEP:
                if not len(cache):
                    cache.append(self._sample())
                    cache.append(self._sample())
                self.__worker_pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            elif op == self.NETWORK:
                self._network = data
            else:
                raise NotImplementedError

    def _transition(self):
        raise NotImplementedError

    def _set_up(self):
        pass

    def step(self):
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        self.__pipe.send([self.EXIT, None])

    def set_network(self, net):
        if not self.config.async_actor:
            self._network = net
        else:
            self.__pipe.send([self.NETWORK, net])