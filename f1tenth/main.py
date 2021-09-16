import os
import time
import yaml
import random
import argparse
from argparse import Namespace
from collections import deque
from datetime import datetime
from itertools import count

import gym
import numpy as np
# from numba import njit
# @njit(fastmath=False, cache=True)

from models.tensorflow_impl import DDPG

parser = argparse.ArgumentParser()
parser.add_argument('--mass', type=float, default=3.463388126201571)
parser.add_argument('--lf', type=float, default=0.15597534362552312)
parser.add_argument('--tlad', type=float, default=0.82461887897713965)
parser.add_argument('--vgain', type=float, default=0.90338203837889)
parser.add_argument('--config', type=str, default='config_example_map.yaml')
parser.add_argument('--batch-size', type=int, default=1024)
args = parser.parse_args()


"""
class DiscountableVariable:

    def __init__(self, initial_value: float):
        self._value = initial_value

    @property
    def value(self):
        # TODO: discount
        return self._value
"""


class ReplayBuffer:

    def __init__(self, maxlen=1_000_000):
        self._memory = deque(maxlen=maxlen)

    def append(self, *args):
        self._memory.append(list(args))

    def extend(self, items):
        self._memory.extend(items)

    def sample(self, size=1024):
        return random.sample(self._memory, k=size)

    def clear(self):
        self._memory.clear()

    def __len__(self):
        return len(self._memory)

    @property
    def items(self): return list(self._memory)


def make_f1tenth_state_vector(obs):
    input_ = [*obs['scans'][0] / 10.0,
              obs['poses_x'][0] / 100.0,
              obs['poses_y'][0] / 100.0,
              np.cos(obs['poses_theta'][0]),
              np.sin(obs['poses_theta'][0]),
              obs['linear_vels_x'][0] / 100.0,
              obs['linear_vels_y'][0] / 100.0,
              obs['ang_vels_z'][0]]
    return np.asarray(input_, dtype=np.float32)


def make_f1tenth_reward(obs):
    # TODO: 정지하는 방향으로 수렴하는 것을 방지하기 위하여 linear_vels_x를 가중치로 이용하는 방법
    return 0.01


def main():
    resource_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    with open(os.path.join(resource_dir, args.config)) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # x(min=-52.14407900642191, max=1.2427659448957173) y(min=-13.919160016844765, max=25.485083751327636)
    env = gym.make('f110_gym:f110-v0', map=os.path.join(resource_dir, conf.map_path), map_ext=conf.map_ext, num_agents=1)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    state = make_f1tenth_state_vector(obs)
    print('state:', state[-1])
    env.render()

    start = time.time()

    agent = DDPG(observation_size=1087, action_size=env.action_space.shape[-1])

    memory = ReplayBuffer()
    tmp_memory = ReplayBuffer()

    reset_required = False
    reward_ = 0

    for i in count(1):
        steer, speed = agent.get_action(state).numpy().squeeze()
        obs, reward, done, info = env.step(np.array([[steer / 100, speed * 10]]))
        next_state = make_f1tenth_state_vector(obs)
        # print(f'[{datetime.now().isoformat()}] Position: {obs["poses_x"][0]}, {obs["poses_y"][0]} / Vel: {obs["linear_vels_x"]}, {obs["linear_vels_y"]}')

        reward_ -= 0.01 + obs['linear_vels_x'][0] / 10000

        if info['lap_passed'][0]:
            print(f'[{datetime.now().isoformat()}] Lap passed! (lap={obs["lap_counts"][0]})')
            # reward = 1.0
            reward_ += 10
        elif done:
            print(f'[{datetime.now().isoformat()}] Collision! (collisions={obs["collisions"][0]})')
            # reward = -1.0
            reward_ -= 10
            reset_required = True
        env.render(mode='human')

        # memory.append(state, (steer, speed), reward, next_state)
        tmp_memory.append(state, (steer, speed), reward_, next_state, done)
        if done or len(tmp_memory) >= args.batch_size:
            gamma = 0.98
            items = tmp_memory.items
            tmp_memory.clear()
            reward = items[-1][2]
            for i in range(len(items)-2, -1, -1):
                reward = gamma * reward * (1 - items[i][-1]) + items[i][2]
                items[i][2] = reward
            memory.extend(items)

        state = next_state.copy()
        if len(memory) >= args.batch_size:
            batch = memory.sample(args.batch_size)
            loss = agent.train(batch)
            print(f'Loss: {loss}')

        if reset_required:
            obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
            reward_ = 0
            reset_required = False

    env.close()


if __name__ == '__main__':
    main()
