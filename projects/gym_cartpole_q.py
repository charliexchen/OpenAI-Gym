import gym, random, copy
import numpy as np
from learn.net import dense_net

from learn.diff_functions import lrelu_fd, sq_diff_fd, sigmoid_fd, tanh_fd, linear_fd

#env = gym.make("LunarLander-v2")
env = gym.make("CartPole-v1")
env.reset()


def geo_sum(discount, i):
    return (1 - discount ** (i)) / (1 - discount)


net = dense_net(5, 512, lrelu_fd, sq_diff_fd, False, 0.00001)
net.add_layer(1, linear_fd, 0.00001)
random_action_prob = 1.0


def next_action(state):
    if random_action_prob < random.uniform(0.0, 1.0):
        pos = net.activate(state + [1])
        neg = net.activate(state + [0])
        return int(0.5 + (0.5 * np.sign(pos - neg)[0]))
    return random.sample([1,0],1)[0]


all_transitions = []


def simulate(steps, gen):
    observation = env.reset()
    states = []
    finished = []
    rewards = []

    trans = []
    for _ in range(steps):
        if random_action_prob < 0 and gen%20==0:
            env.render()
        action = next_action(list(observation))
        states.append(list(copy.deepcopy(observation)) + [action])
        observation, reward, done, info = env.step(action)
        all_transitions.append(
            {"input": states[-1], "obs": copy.deepcopy(observation), "reward": 1.0-observation[2]**2-0.1*observation[0]**2})

        finished.append(done)

        if done:
            observation = env.reset()
    print("generation", gen, "deaths", sum(1 if fin else 0 for fin in finished), "prob",
          random_action_prob)
    return states, rewards
    count = 0
    for i in range(len(finished))[::-1]:
        bkup = finished[i]
        count += 1
        if finished[i]:
            count = 0
        finished[i] = geo_sum(discount, count + 1) - 1

    for i, val in enumerate(finished):
        trans[i]["reward"] = val
    all_transitions.extend(trans)
    return states, finished


discount = 0.85
for i in range(10000):
    random_action_prob -= 0.01
    states, rewards = simulate(2000, i)
    predicted = []
    actual = []
    samples = random.sample(all_transitions, 2000)
    for sample in samples:
        predicted.append(
            sample["reward"] + discount * max(
                net.activate(list(sample["obs"]) + [1]),
                net.activate(list(sample["obs"]) + [0])
            )
        )
        actual.append(sample["input"])
    print("fitness:", net.update(actual, predicted, a=-0.1/(i+1)))





env.close()
