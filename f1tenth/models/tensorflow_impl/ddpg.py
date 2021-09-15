import tensorflow as tf
import numpy as np


"""
class ActorCritic(tf.keras.Model):

    def __init__(self, observation_size, action_size):
        super(ActorCritic, self).__init__()
        self._observation_size = observation_size

        self.dense1 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform')
        self.actor = tf.keras.layers.Dense(action_size, activation='tanh', kernel_initializer='glorot_uniform')
        self.critic = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer='glorot_uniform')

    def call(self, x):
        x = tf.convert_to_tensor(x)
        x = tf.reshape(x, [-1, self.observation_size])
        x = self.dense1(x)
        x = self.dense2(x)
        policy = self.actor(x)
        value = self.critic(tf.concat([x, policy], axis=-1))
        return policy, value

    @property
    def observation_size(self): return self._observation_size
"""


class Actor(tf.keras.Model):

    def __init__(self, observation_size, action_size):
        super(Actor, self).__init__()
        self._observation_size = observation_size

        self.dense1 = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_uniform')
        self.dense2 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')
        self.actor = tf.keras.layers.Dense(action_size, activation='tanh', kernel_initializer='glorot_uniform')

    def call(self, x):
        x = tf.convert_to_tensor(x)
        x = tf.reshape(x, [-1, self.observation_size])
        x = self.dense1(x)
        x = self.dense2(x)
        policy = self.actor(x)
        return policy

    @property
    def observation_size(self): return self._observation_size


class Critic(tf.keras.Model):

    def __init__(self, observation_size, action_size):
        super(Critic, self).__init__()
        self._observation_size = observation_size
        self._action_size = action_size

        self.dense1 = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_uniform')
        self.dense2 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')
        self.critic = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer='glorot_uniform')

    def call(self, s, a):
        s = tf.convert_to_tensor(s)
        s = tf.reshape(s, [-1, self.observation_size])
        a = tf.convert_to_tensor(a)
        a = tf.reshape(a, [-1, self.action_size])
        x = tf.concat([s, a], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        value = self.critic(x)
        return value

    @property
    def observation_size(self): return self._observation_size

    @property
    def action_size(self): return self._action_size


class DDPG:

    def __init__(self, observation_size, action_size, learning_rate=1e-4, gamma=0.99, tau=1e-3):
        super(DDPG, self).__init__()
        self._observation_size = observation_size
        self._action_size = action_size
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._tau = tau

        # self.model = ActorCritic(observation_size, action_size)
        # self.target_network = ActorCritic(observation_size, action_size)
        # self.target_network.set_weights(self.model.get_weights())
        self.actor = Actor(observation_size, action_size)
        self.target_actor = Actor(observation_size, action_size)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic = Critic(observation_size, action_size)
        self.target_critic = Critic(observation_size, action_size)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
    def get_action(self, states):
        # TODO: Noise (https://openai.com/blog/better-exploration-with-parameter-noise/)
        # policy, _ = self.model(states)
        # return policy
        return self.actor(states)

    def get_value(self, states, actions):
        # _, value = self.model(states)
        # return value
        return self.critic(states, actions)

    def train(self, batch):
        states = []
        actions = []
        rewards = []
        next_states = []
        for tuple_ in batch:
            states.append(tuple_[0])
            actions.append(tuple_[1])
            rewards.append(tuple_[2])
            next_states.append(tuple_[3])
        s = np.asarray(states, dtype=np.float32)
        a = np.asarray(actions, dtype=np.float32)
        r = np.asarray(rewards, dtype=np.float32)
        s_next = np.asarray(next_states, dtype=np.float32)

        critic_loss = self._get_critic_loss(s, a, r, s_next)
        actor_loss = self._get_actor_loss(s, a, r, s_next)

        self.update_target_network(self.actor, self.target_actor)
        self.update_target_network(self.critic, self.target_critic)

        return actor_loss.sum() + critic_loss.sum()

    def _get_actor_loss(self, s, a, r, s_next):
        with tf.GradientTape() as tape:
            actions = self.actor(s)
            # loss = -self.critic(s, actions)
            loss = self.critic(s, actions)
        grads = tape.gradient(loss, self.actor.trainable_variables)
        # grads = tf.clip_by_value(grads, clip_value_min=-20.0, clip_value_max=20.0)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        return loss.numpy()

    def _get_critic_loss(self, s, a, r, s_next):
        with tf.GradientTape() as tape:
            target_next_action = self.target_actor(s_next)
            target_next_q = self.target_critic(s_next, target_next_action)
            target_q = r + self.gamma * target_next_q

            current_q = self.critic(s, a)
            loss = tf.keras.losses.huber(target_q, current_q)

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        return loss.numpy()

    def update_target_network(self, model, target_network):
        for w, target_w in zip(model.weights, target_network.weights):
            target_w.assign(self.tau * w + (1 - self.tau) * target_w)

    @property
    def observation_size(self): return self._observation_size

    @property
    def action_size(self): return self._action_size

    @property
    def learning_rate(self): return self._learning_rate

    @property
    def gamma(self): return self._gamma

    @property
    def tau(self): return self._tau


if __name__ == "__main__":
    batch_size = 4
    observation_size = 16
    action_size = 2
    x = np.random.uniform(-1.0, 1.0, (batch_size, observation_size))
    x = tf.convert_to_tensor(x)

    agent = DDPG(observation_size, action_size)
    action = agent.get_action(x)
    value = agent.get_value(x, action)

    print('observation:', x)
    print('action:', action)
    print('value:', value)
