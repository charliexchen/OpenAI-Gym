import tensorflow as tf
import keras
import numpy as np
import gym
import random
import time

from collections import deque


class doubleQLearner():
    state_size = 8
    action_count = 4

    def __init__(self,
                 learning_rate=0.0005,
                 discount=0.9):
        self.learning_rate = learning_rate
        self.discount = discount
        self.Q1 = self.build_network()
        self.Q2 = self.build_network()
        self.rand_action = 1.0
        self.rand_action_decay = 0.996
        self.rand_action_min = 0.05

        self.memory = deque(maxlen=1000000)

    def store_sample(self, item):

        self.memory.append(item)

    def get_samples(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        actions, after_states, rewards, states = [], [], [], []
        for samples in batch:
            action, after_state, reward, state = samples
            actions.append(action)
            after_states.append(after_state)
            rewards.append(reward)
            states.append(state)

        return (np.array(actions), np.array(after_states), np.array(rewards),
                np.array(states))

    def sample_train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        actions, after_states, rewards, states = self.get_samples(batch_size)
        self._learn_single_model(self.Q1, self.Q1, states,
                                 rewards,
                                 after_states, actions)
        #actions, after_states, rewards, states = self.get_samples(batch_size)
        #self._learn_single_model(self.Q1, self.Q2, states,
        #                         rewards,
        #                         after_states, actions)
        if self.rand_action > self.rand_action_min:
            self.rand_action *= self.rand_action_decay

    def build_network(self):
        model = keras.Sequential([
            keras.layers.Dense(512, input_dim=self.state_size,
                               activation='relu'),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(self.action_count, activation='linear')
        ])

        opt = keras.optimizers.adam(lr=self.learning_rate)
        model.compile(optimizer=opt,
                      loss='mse',
                      )
        return model

    def evaluate_state_actions(self, states):
        q1_value = self.Q1.predict(states)
        #q2_value = self.Q2.predict(states)
        return np.argmax(q1_value , axis=1)

    def policy(self, state, random_probability=None):
        if not random_probability:
            random_probability = self.rand_action
        roll = np.random.random()
        if roll < random_probability:
            return np.random.randint(self.action_count)
        else:
            return self.evaluate_state_actions(state)[0]

    def _learn_single_model(self, learning_model, bootstrapping_model, states,
                            rewards, after_states, actions):

        bootstrap_action_selection_values = learning_model.predict_on_batch(
            after_states)
        bootstrap_action_values = bootstrapping_model.predict_on_batch(after_states)

        bootstrap_actions = np.argmax(bootstrap_action_selection_values, axis=1)
        bootstrap_actions_1h = keras.utils.to_categorical(bootstrap_actions,
                                                          self.action_count)
        td_action_values = rewards + self.discount * (
            np.sum(bootstrap_actions_1h * bootstrap_action_values, axis=1))
        current_values = learning_model.predict_on_batch(states)

        for i in range(len(states)):
            current_values[i][actions[i]] = td_action_values[i]

        learning_model.fit(states, current_values, epochs=1, verbose=0)

    def learn(self, states, rewards, after_states, actions):
        # movefast -- screw good code
        assert len(rewards) == len(states)
        assert len(after_states) == len(states)
        states1, states2 = [], []
        rewards1, rewards2 = [], []
        actions1, actions2 = [], []
        after_states1, after_states2 = [], []
        for i, x in enumerate(
                np.random.randint(2, size=len(states))):
            if x == 1:
                states1.append(states[i])
                rewards1.append(rewards[i])
                actions1.append(actions[i])
                after_states1.append(after_states[i])
            else:
                states2.append(states[i])
                rewards2.append(rewards[i])
                actions2.append(actions[i])
                after_states2.append(after_states[i])
        if len(states1) > 0:
            self._learn_single_model(self.Q2, self.Q1, np.array(states1),
                                     np.array(rewards1),
                                     np.array(after_states1), np.array(actions1))
        if len(states2) > 0:
            self._learn_single_model(self.Q1, self.Q2, np.array(states2),
                                     np.array(rewards2),
                                     np.array(after_states2), np.array(actions2))


if __name__ == "__main__":
    policy = doubleQLearner()

    generation = 0
    env = gym.make('LunarLander-v2')
    env = env.unwrapped

    done = False
    episode_count = 0
    episode_len = 0
    state = env.reset()
    episode_reward = 0
    env.seed(0)

    np.random.seed(0)
    while True:
        if generation % 1 == 0:
            env.render()
        action = policy.policy(np.array([state]))
        state_, reward, done, info = env.step(action)
        data = (action, state_, reward, state)
        policy.store_sample(data)
        state = state_
        episode_len += 1
        episode_reward += reward
        if episode_len > 3000:
            done = True
            state = env.reset()
        if done:
            episode_count += 1
            episode_len = 0
            state = env.reset()
            generation += 1

            print(
                'Episode :' + str(episode_count) + ' === Reward: ' + str(
                    episode_reward))
            episode_reward = 0
            if policy.discount<0.99:
                policy.discount+=0.001*0.09
        policy.sample_train()
