import gym, random, copy
import numpy as np
from learn.net import dense_net



from learn.diff_functions import lrelu_fd, log_pol_fd, softmax_fd, sigmoid_fd

env = gym.make("LunarLander-v2")

net = dense_net(8, 16, lrelu_fd, log_pol_fd, False, 0.01)
net.add_layer(16, lrelu_fd, 0.01)
net.add_layer(4, softmax_fd, 0.01)


def next_action(observation, sampled=True, input_net = net):
    distribution = input_net.activate(observation)
    if sampled:
        action = np.random.choice(range(len(distribution)), p = distribution)
    else:
        action = np.argmax(distribution)
    return action


def gen_trajectories(trajs=10, maxlen=500, render=False):
    trajectories = []
    for _ in range(trajs):
        observation = env.reset()
        new_trajectory = {'state': [], 'reward': [], 'action': [], 'next_state': []}
        for __ in range(maxlen):
            new_trajectory['state'].append(observation)
            action = next_action(list(observation))
            observation, reward, done, info = env.step(action)
            new_trajectory['reward'].append(
                reward# - observation[2] ** 2 - observation[0] ** 2
            )

            new_trajectory['action'].append(int(action))
            new_trajectory['next_state'].append(observation)
            if done:
                break
        all_reward = sum(new_trajectory['reward'])
        new_trajectory['total_reward'] = all_reward
        new_trajectory['reward'] = [x/abs(all_reward) for x in
                                    new_trajectory['reward']]
        trajectories.append(new_trajectory)
    return trajectories


def display_episode():
    observation = env.reset()
    done = False
    while not done:
        action = next_action(list(observation))
        observation, reward, done, info = env.step(action)
        env.render()
    env.reset()

'''
print("Searching favorable initial conditions...")
for i in range(5):
    for j in range(10):
        new_net = copy.deepcopy(net)
        new_net.mutate(0.1)
        fitness = 0
        observation = env.reset()
        for _ in range(500):
            action = next_action(list(observation), input_net = new_net)
            observation, reward, done, info = env.step(action)
            fitness+=reward
            if done:
                break
        if i==0:
            max_fitness = fitness
        else:
            if fitness > max_fitness:
                max_fitness = fitness
                net = new_net
    print(max_fitness)

'''

discount = 0.95
for generation in range(100000):
    #if generation % 1 == 0:
    display_episode()
    trajectories = gen_trajectories()
    average_reward = 0
    for trajectory in trajectories:
        #net.update(trajectory['state'], trajectory['action'],
        #           sum(trajectory['reward']) * 0.02 / (generation*0.01 + 1))
        net.update_trajectory(trajectory, discount, 0.02/(generation*0.01+1))
        average_reward+=trajectory['total_reward']
    average_reward = average_reward/len(trajectories)
    print(
        f"generation {generation}: average trajectory total reward {average_reward}")


def update_net():
    pass


env.close()
