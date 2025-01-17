# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Author: Hongrui Zheng
'''
import os
import time

# gym imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# base classes
from f110_gym.envs.base_classes import Simulator

# others
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

# constants

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800


class F110Env(gym.Env, utils.EzPickle):
    """
    OpenAI gym environment for F1TENTH

    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility

            map (str, default='vegas'): name of the map used for the environment. Currently, available environments include: 'berlin', 'vegas', 'skirk'. You could use a string of the absolute path to the yaml file of your custom map.

            map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'

            params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front
            C_Sr: Cornering stiffness coefficient, rear
            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=2): number of agents in the environment

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
            max_lap (int, default=inf)
    """
    metadata = {'render.modes': ['human', 'human_fast']}

    def __init__(self, **kwargs):
        # kwargs extraction
        try:
            self.seed = kwargs['seed']
        except:
            self.seed = 12345
        try:
            self.map_name = kwargs['map']
            # different default maps
            if self.map_name == 'berlin':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/berlin.yaml'
            elif self.map_name == 'skirk':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/skirk.yaml'
            elif self.map_name == 'levine':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/levine.yaml'
            else:
                self.map_path = self.map_name + '.yaml'
        except:
            self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/vegas.yaml'

        try:
            self.map_ext = kwargs['map_ext']
        except:
            self.map_ext = '.png'

        try:
            self.params = kwargs['params']
        except:
            self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

        # simulation parameters
        try:
            self.num_agents = kwargs['num_agents']
        except:
            self.num_agents = 2

        try:
            self.timestep = kwargs['timestep']
        except:
            self.timestep = 0.01

        # default ego index
        try:
            self.ego_idx = kwargs['ego_idx']
        except:
            self.ego_idx = 0

        self.max_lap = kwargs.get('max_lap', float('inf'))

        self.starting_point = kwargs.get('starting_point', np.zeros((self.num_agents, 3)))

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents, ))
        # TODO: collision_idx not used yet
        # self.collision_idx = -1 * np.ones((self.num_agents, ))

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents, ))
        self.lap_counts = np.zeros((self.num_agents, ))
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.start_xs = np.zeros((self.num_agents, ))
        self.start_ys = np.zeros((self.num_agents, ))
        self.start_thetas = np.zeros((self.num_agents, ))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(self.params, self.num_agents, self.seed)
        self.sim.set_map(self.map_path, self.map_ext)
        self.map_center = np.array([-28, 0])

        # rendering
        self.renderer = None
        self.current_obs = None

        """
        self._observation_space = gym.spaces.Dict({
            'edo_idx': gym.spaces.Discrete(1),
            'scans': gym.spaces.Box(0.0, 32.0, shape=(self.num_agents, 1080)),
            'poses_x': gym.spaces.Box(-100.0, 100.0, shape=(self.num_agents,)),
            'poses_y': gym.spaces.Box(-100.0, 100.0, shape=(self.num_agents,)),
            'poses_theta': gym.spaces.Box(0.0, np.pi * 2, shape=(self.num_agents,)),
            'linear_vels_x': gym.spaces.Box(-1.0, 1.0, shape=(self.num_agents,)),
            'linear_vels_y': gym.spaces.Box(-1.0, 1.0, shape=(self.num_agents,)),
            'ang_vels_z': gym.spaces.Box(0.0, np.pi * 2, shape=(self.num_agents,)),
            'collisions': gym.spaces.Box(0.0, float('inf'), shape=(self.num_agents,), dtype=np.uint8),
            'lap_times': gym.spaces.Box(0.0, float('inf'), shape=(self.num_agents,)),
            'lap_counts': gym.spaces.Box(0.0, float('inf'), shape=(self.num_agents,), dtype=np.uint8)
        })
        """
        self._observation_space = gym.spaces.Box(-1.0, 1.0, shape=(self.num_agents, 1086))
        self._action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.num_agents, 2))

        self._checkpoint_reaches = np.zeros((self.num_agents, 3))   # .astype(np.bool)

        self._frame_dir = os.path.join(os.path.dirname(__file__), 'tmp')
        if not os.path.exists(self._frame_dir):
            os.mkdir(self._frame_dir)

    def __del__(self):
        """
        Finalizer, does cleanup
        """
        pass

    def _check_done(self):
        """
        Check if the current rollout is done

        Args:
            None

        Returns:
            done (bool): whether the rollout is done
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t = 2
        right_t = 2

        poses_x = np.array(self.poses_x)-self.start_xs
        poses_y = np.array(self.poses_y)-self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_x = delta_pt[0, :]
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        closes = (temp_x ** 2 + temp_y ** 2) <= 0.1
        self._update_checkpoint_reaches(temp_x, temp_y)
        for i in range(self.num_agents):
            if closes[i] and temp_y[i] >= 0 and np.all(self._checkpoint_reaches[i]):
                self.lap_counts[i] += 1
                self.lap_times[i] = self.current_time
                self._checkpoint_reaches[i, :] = False

        done = (self.collisions[self.ego_idx]) or (self.lap_counts > self.last_lap_counts)[self.ego_idx]

        return done

    def _update_checkpoint_reaches(self, temp_x, temp_y):
        for i in range(self.num_agents):
            if not self._checkpoint_reaches[i][0]:
                if temp_x[i] > 10 and temp_y[i] >= 25:
                    self._checkpoint_reaches[i][0] = True
                continue
            elif not self._checkpoint_reaches[i][1]:
                if np.abs(temp_x[i]) < 1.0 and temp_y[i] >= 30:
                    self._checkpoint_reaches[i][1] = True
                continue
            elif not self._checkpoint_reaches[i][2]:
                if temp_x[i] < -4 and temp_y[i] <= 25:
                    self._checkpoint_reaches[i][2] = True

    def _update_state(self, obs_dict):
        """
        Update the env's states according to observations

        Args:
            obs_dict (dict): dictionary of observation

        Returns:
            None
        """
        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.collisions = obs_dict['collisions']

    def _get_angle_on_track(self):
        pos = np.concatenate([self.poses_x, self.poses_y], axis=-1)
        dx, dy = pos - self.map_center
        degree = np.arctan2(dy, dx) * 180 / np.pi
        if degree < 0:
            degree = 360 + degree
        return degree

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """

        # call simulation step
        obs = self.sim.step(action)
        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts

        self.current_obs = obs

        # update data member
        self._update_state(obs)

        # times
        track_angle = self._get_angle_on_track()
        reward = track_angle / 360 - 1
        # reward = self.timestep
        self.current_time = self.current_time + self.timestep

        # check done
        done = self._check_done()
        lap_passed = (self.lap_counts > self.last_lap_counts).tolist()
        info = {'lap_passed': lap_passed}

        self.last_lap_counts[:] = self.lap_counts

        if self.renderer is not None:
            self.renderer.save_frame(os.path.join(self._frame_dir, f'{self._frame_count:08d}.png'))
            self._frame_count += 1

        # FIXME: obs
        obs = np.asarray([*obs['scans'][self.ego_idx] / 10.0,
                          obs['poses_x'][self.ego_idx] / 50.0,
                          obs['poses_y'][self.ego_idx] / 50.0,
                          np.cos(obs['poses_theta'][self.ego_idx]),
                          np.sin(obs['poses_theta'][self.ego_idx]),
                          obs['linear_vels_x'][self.ego_idx] / 10.0,
                          obs['ang_vels_z'][self.ego_idx]],
                         dtype=np.float32)

        return obs, reward, done, info

    def reset(self, poses=None):
        """
        Reset the gym environment by given poses

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        self._frame_count = 0
        if self.renderer is not None:
            self.gif()
            for file_ in os.listdir(self._frame_dir):
                os.remove(os.path.join(self._frame_dir, file_))

        poses = poses or self.starting_point

        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents, ))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.lap_counts - np.zeros((self.num_agents,))
        self.last_lap_counts = self.lap_counts.copy()

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array([[np.cos(-self.start_thetas[self.ego_idx]), -np.sin(-self.start_thetas[self.ego_idx])], [np.sin(-self.start_thetas[self.ego_idx]), np.cos(-self.start_thetas[self.ego_idx])]])

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, _, _, _ = self.step(action)
        return obs

    def update_map(self, map_path, map_ext):
        """
        Updates the map used by simulation

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file

        Returns:
            None
        """
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles

        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def render(self, mode='human'):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        assert mode in ['human', 'human_fast']
        if self.renderer is None:
            # first call, initialize everything
            from f110_gym.envs.rendering import EnvRenderer
            self.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            self.renderer.update_map(self.map_name, self.map_ext)
        self.renderer.update_obs(self.current_obs)
        self.renderer.dispatch_events()
        self.renderer.on_draw()
        self.renderer.flip()
        if mode == 'human':
            time.sleep(0.005)
        elif mode == 'human_fast':
            pass

    def gif(self):
        fig = plt.figure()
        # paths = os.listdir(self._frame_dir)
        # artists = tuple(map(lambda x: Image.open(os.path.join(self._frame_dir, x)), paths))
        artists = []
        for path in os.listdir(self._frame_dir):
            image = Image.open(os.path.join(self._frame_dir, path))
            artists.append([plt.imshow(image, animated=True)])
        anim = animation.ArtistAnimation(fig, artists, interval=int(1000/60), repeat=True)
        anim.save(os.path.join(os.path.dirname(__file__), f'{int(time.time() * 1000)}.gif'))

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


if __name__ == "__main__":
    pass
