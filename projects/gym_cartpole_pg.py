import gym, random, copy
import numpy as np
from learn.net import dense_net

from learn.diff_functions import lrelu_fd, log_pol_fd, softmax_fd

env = gym.make("CartPole-v1")
#env = gym.make("LunarLander-v2")

net = dense_net(4, 128, lrelu_fd, log_pol_fd, False, 0.00001)
net.add_layer(2, softmax_fd, 0.00001)


def next_action(observation, sampled=True):
    distribution = net.activate(observation)
    if sampled:
        return np.random.choice(range(len(distribution)), p = distribution)
    else:
        return 0 if distribution[0] > distribution[1] else 1


def gen_trajectories(trajs=100, maxlen=500, render=False):
    trajectories = []
    for _ in range(trajs):
        observation = env.reset()
        new_trajectory = {'state': [], 'reward': [], 'action': [], 'next_state': []}
        for __ in range(maxlen):
            new_trajectory['state'].append(observation)
            action = next_action(list(observation))
            observation, reward, done, info = env.step(action)
            new_trajectory['reward'].append(
                1.0# - observation[2] ** 2 - observation[0] ** 2
            )
            new_trajectory['action'].append(int(action))
            new_trajectory['next_state'].append(observation)
            if done:
                break
        trajectories.append(new_trajectory)
    return trajectories


def display_episode():
    observation = env.reset()
    done = False
    while not done:
        action = next_action(list(observation), sampled=False)
        observation, reward, done, info = env.step(action)
        env.render()
    env.reset()

discount = 0.9
for generation in range(100000):
    trajectories = gen_trajectories()
    average_reward = 0
    for trajectory in trajectories:
        #net.update(trajectory['state'],trajectory['action'],sum(trajectory['reward'])*0.001/(generation+1))
        net.update_trajectory(trajectory, discount, 0.001/(generation+1))
        average_reward+=sum(trajectory['reward'])
    average_reward = average_reward/len(trajectories)
    if generation % 1==0:
        print(f"generation {generation}: average trajectory total reward {average_reward}")
        display_episode()


def update_net():
    pass


env.close()
