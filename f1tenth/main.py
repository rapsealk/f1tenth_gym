import os
import yaml
import argparse
from argparse import Namespace

import gym
import numpy as np
# from numba import njit
# @njit(fastmath=False, cache=True)

parser = argparse.ArgumentParser()
parser.add_argument('--mass', type=float, default=3.463388126201571)
parser.add_argument('--lf', type=float, default=0.15597534362552312)
parser.add_argument('--tlad', type=float, default=0.82461887897713965)
parser.add_argument('--vgain', type=float, default=0.90338203837889)
parser.add_argument('--config', type=str, default='config_example_map.yaml')
parser.add_argument('--batch-size', type=int, default=64)
args = parser.parse_args()


def make_f1tenth_state_vector(obs):
    input_ = [*obs['scans'][0] / 10.0,
              obs['poses_x'][0] / 50.0,
              obs['poses_y'][0] / 50.0,
              np.cos(obs['poses_theta'][0]),
              np.sin(obs['poses_theta'][0]),
              obs['linear_vels_x'][0] / 10.0,
              # obs['linear_vels_y'][0] / 100.0,
              obs['ang_vels_z'][0]]
    return np.asarray(input_, dtype=np.float32)


def merge_state_vector(prev, new):
    return np.stack([*prev[1:], new])


def main():
    resource_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    with open(os.path.join(resource_dir, args.config)) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # x(min=-52.14407900642191, max=1.2427659448957173) y(min=-13.919160016844765, max=25.485083751327636)
    env = gym.make('f110_gym:f110-v0', map=os.path.join(resource_dir, conf.map_path), map_ext=conf.map_ext, num_agents=1,
                   starting_point=np.array([[conf.sx, conf.sy, conf.stheta]]))

    from models.tensorflow_impl import DDPGAgent

    agent = DDPGAgent(action_size=env.action_space.shape[-1])
    agent.train(env)
    return


if __name__ == '__main__':
    main()
