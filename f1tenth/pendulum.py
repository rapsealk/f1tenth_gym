import argparse
from collections import namedtuple
from itertools import count

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='Pendulum-v0')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--std-dev', type=float, default=0.2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.005)

# History = namedtuple('History', ('obs', 'action', 'reward', 'next_obs'))


class OrnsteinUhlenbeckActionNoise:

    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_dev
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (self.x_prev
             + self.theta * (self.mean - self.x_prev) * self.dt
             + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Actor(tf.keras.Model):

    def __init__(self):
        super(Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform')
        self.actor = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.actor(x)


class Critic(tf.keras.Model):

    def __init__(self):
        super(Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, activation='relu', kernel_initializer='he_uniform')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.action_dense = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.concat = tf.keras.layers.Concatenate()
        self.concat_dense1 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform')
        self.concat_dense2 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform')
        self.critic = tf.keras.layers.Dense(1)

    def call(self, s, a):
        s = self.dense1(s)
        s = self.dense2(s)
        a = self.action_dense(a)
        s = self.concat([s, a])
        s = self.concat_dense1(s)
        s = self.concat_dense2(s)
        return self.critic(s)


class Agent:

    def __init__(self, lower_bound, upper_bound,
                 actor_learning_rate=0.001,
                 critic_learning_rate=0.002,
                 std_dev=0.2,
                 gamma=0.99,
                 tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(1), std_dev=float(std_dev) * np.ones(1))

        self.actor = Actor()
        self.critic = Critic()

        self.target_actor = Actor()
        self.target_critic = Critic()

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)

    def get_action(self, state):
        sampled_actions = tf.squeeze(self.actor(state)).numpy() + self.noise()
        legal_actions = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        return [legal_actions.squeeze()]

    # @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch)
            y = reward_batch + self.gamma * self.target_critic(next_state_batch, target_actions)
            critic_value = self.critic(state_batch, action_batch)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch)
            critic_value = self.critic(state_batch, actions)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self._update_target(self.target_actor.variables, self.actor.variables)
        self._update_target(self.target_critic.variables, self.critic.variables)

        return critic_loss.numpy().sum() + actor_loss.numpy().sum()

    @tf.function
    def _update_target(self, target_weights, weights):
        for target_w, w in zip(target_weights, weights):
            target_w.assign(w * self.tau + target_w * (1 - self.tau))


class Buffer:

    def __init__(self, num_states, num_actions, buffer_capacity=100_000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter = min(self.buffer_counter + 1, self.buffer_capacity)

    def sample(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.cast(
            tf.convert_to_tensor(self.reward_buffer[batch_indices]),
            dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        return (state_batch, action_batch, reward_batch, next_state_batch)

    def __len__(self):
        return self.buffer_counter


def main():
    from models.tensorflow_impl import DDPGAgent

    env = gym.make('Pendulum-v0')
    agent = DDPGAgent(action_size=env.action_space.shape[-1])
    agent.train(env)
    return

    args = parser.parse_args()
    env = gym.make(args.env)

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    agent = Agent(lower_bound=lower_bound, upper_bound=upper_bound, std_dev=args.std_dev, gamma=args.gamma, tau=args.tau)
    buffer = Buffer(num_states=env.observation_space.shape[0], num_actions=env.action_space.shape[0], batch_size=args.batch_size)

    episode_rewards = []
    average_rewards = []

    for episode in count(1):
        state = env.reset()
        episodic_reward = 0

        while True:
            env.render()
            state_tf = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
            action = agent.get_action(state_tf)
            next_state, reward, done, info = env.step(action)

            buffer.record((state, action, reward, next_state))
            episodic_reward += reward

            if len(buffer) >= args.batch_size:
                batch = buffer.sample()
                loss = agent.update(*batch)
                # print(f'Loss: {loss}')

            if done:
                break

            state = next_state

        episode_rewards.append(episodic_reward)
        average_rewards.append(np.mean(episode_rewards[-40:]))
        print(f'Episode #{episode} Average reward: {average_rewards[-1]:.4f}')


if __name__ == "__main__":
    main()
