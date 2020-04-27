import pygame
import numpy as np
import pickle, math, random
from learn.net import dense_net
from learn.diff_functions import lrelu_fd, sq_diff_fd, sigmoid_fd, tanh_fd, linear_fd

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

boundary = [500, 500]


class point:
    def __init__(self, pos, vel, mass, origin):
        self.origin = origin
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.acc = (0, 0)

    def rad(self):
        return int(5 * math.sqrt(self.mass))

    def ppos(self):
        return tuple(int(x) for x in self.pos + self.origin)


class double_pendulum:
    def __init__(self, params):
        self.params = params
        self.reset()

    def reset(self):
        self.p1 = point(self.params['x1'], self.params['v1'], self.params['m1'], self.params['origin'])
        self.p2 = point(self.params['x2'], self.params['v2'], self.params['m2'], self.params['origin'])
        self.r1 = np.linalg.norm(self.params['x1'])
        self.r2 = np.linalg.norm(self.params['x1'] - self.params['x2'])
        if random.choice([1, 0]) == 0:
            self.p1.pos[0] = -self.p1.pos[0]
            self.p2.pos[0] = -self.p2.pos[0]

    def tension(self):
        m1 = self.params['m1']
        m2 = self.params['m2']
        w1 = self.p1.pos
        dw1 = self.p1.vel
        w2 = self.p2.pos - self.p1.pos
        dw2 = self.p2.vel - self.p1.vel
        g = self.params['g']
        b = np.array([
            -m1 * np.dot(dw1, dw1) - m1 * np.dot(g, w1),
            -np.dot(dw2, dw2),
            0.,
            0.
        ])

        A = np.array([
            [w1[0], w1[1], -w1[0], -w1[1]],
            [-w2[0] / m1, -w2[1] / m1, w2[0] * (1 / m1 + 1 / m2),
             w2[1] * (1 / m1 + 1 / m2)],
            [-w1[1], w1[0], 0, 0],
            [0., 0., -w2[1], w2[0]]
        ])

        T = np.linalg.solve(A, b)

        return np.array([T[0], T[1]]), np.array([T[2], T[3]])

    def acc(self):
        g = self.params['g']
        m1 = self.params['m1']
        m2 = self.params['m2']
        T1, T2 = self.tension()
        a1 = (T1 - T2) / m1 + g
        a2 = T2 / m2 + g
        return a1, a2

    def update(self):
        h = 1.0 / self.params['framerate']
        a1, a2 = self.acc()

        self.p1.pos += h * self.p1.vel
        self.p2.pos += h * self.p2.vel
        self.p1.vel += h * a1
        self.p2.vel += h * a2

    def update_cons(self):
        h = 1 / self.params['framerate']
        a1, a2 = self.acc()

        self.p1.pos += h * self.p1.vel
        self.p2.pos += h * self.p2.vel
        self.p1.vel += h * a1
        self.p2.vel += h * a2

        x1 = self.p1.pos
        x2 = self.p2.pos
        v1 = self.p1.vel
        v2 = self.p2.vel
        r1 = self.r1
        r2 = self.r2

        w2 = x2 - x1
        dw2 = v2 - v1

        self.p1.pos = x1 * r1 / np.linalg.norm(x1)
        self.p1.vel = v1 - x2 * np.dot(v1, x1) / (np.dot(x1, x1))
        self.p2.pos = x1 + w2 * r2 / np.linalg.norm(w2)
        self.p2.vel = v1 + dw2 - w2 * np.dot(w2, dw2) / (np.dot(w2, w2))

    def f(self, vec, t=0):
        g = self.params['g']
        m1 = self.params['m1']
        m2 = self.params['m2']

        x1, x2, v1, v2 = vec.reshape(4, 2)
        w1 = x1
        dw1 = v1
        w2 = x2 - x1
        dw2 = v2 - v1

        F = t * np.array([-w1[1], w1[0]])

        b = np.array([
            -m1 * np.dot(dw1, dw1) - m1 * np.dot(g, w1),
            -np.dot(dw2, dw2) + np.dot(F, w2) / m1,
            0.,
            0.
        ])

        A = np.array([
            [w1[0], w1[1], -w1[0], -w1[1]],
            [-w2[0] / m1, -w2[1] / m1, w2[0] * (1 / m1 + 1 / m2),
             w2[1] * (1 / m1 + 1 / m2)],
            [-w1[1], w1[0], 0, 0],
            [0., 0., -w2[1], w2[0]]
        ])

        T = np.linalg.solve(A, b)
        T1, T2 = T.reshape(2, 2)

        a1 = (T1 - T2) / m1 + g
        a2 = T2 / m2 + g
        return np.concatenate((v1, v2, a1, a2), axis=None)

    def rk_step(self, init, t):
        h = 1 / self.params['framerate']
        k1 = h * self.f(init, t)
        k2 = h * self.f(init + k1 / 2, t)
        k3 = h * self.f(init + k2 / 2, t)
        k4 = h * self.f(init + k3, t)
        return init + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def update_rk(self, t):
        init = np.concatenate((self.p1.pos, self.p2.pos, self.p1.vel, self.p2.vel),
                              axis=None)
        inc = self.rk_step(init, t)

        self.p1.pos, self.p2.pos, self.p1.vel, self.p2.vel = inc.reshape(4, 2)

    def update_rk_cons(self, t):
        init = np.concatenate((self.p1.pos, self.p2.pos, self.p1.vel, self.p2.vel),
                              axis=None)
        inc = self.rk_step(init, t)
        self.p1.pos, self.p2.pos, self.p1.vel, self.p2.vel = inc.reshape(4, 2)

        x1 = self.p1.pos
        x2 = self.p2.pos
        v1 = self.p1.vel
        v2 = self.p2.vel
        r1 = self.r1
        r2 = self.r2

        w2 = x2 - x1
        dw2 = v2 - v1

        self.p1.pos = x1 * r1 / np.linalg.norm(x1)
        self.p1.vel = v1 - x2 * np.dot(v1, x1) / (np.dot(x1, x1))
        self.p2.pos = x1 + w2 * r2 / np.linalg.norm(w2)
        self.p2.vel = v1 + dw2 - w2 * np.dot(w2, dw2) / (np.dot(w2, w2))

    def energy(self):
        m1 = self.params['m1']
        m2 = self.params['m2']
        g = self.params['g']
        x1 = self.p1.pos
        x2 = self.p2.pos
        v1 = self.p1.vel
        v2 = self.p2.vel
        pot = - np.dot(g, m1 * x1 + m2 * x2)
        kin = 0.5 * (m1 * np.dot(v1, v1) + m2 * np.dot(v2, v2))
        return pot + kin

    def state_2d(self):
        x, y = self.p1.pos
        dx, dy = self.p1.vel
        theta = math.atan2(x, -y)
        dtheta = -(x * dy - dx * y) / (x ** 2 + y ** 2)
        return [theta, dtheta]

    def reward_2d(self):
        return -self.p1.pos[1]

    @staticmethod
    def discounted_reward(raw_reward, y=0):
        current_reward = 0
        discounted_rewards = []
        for reward in raw_reward[::-1]:
            current_reward += reward
            discounted_rewards.append(current_reward)
            current_reward *= y
        return discounted_rewards[::-1]

    def gen_trajectory_2c(self, max_len=300, sampling=10, f=None, y=0):
        if f == None:
            f = lambda x: 0
        actions = []
        rewards = []
        states = []
        for i in range(max_len):
            if i % sampling == 0:
                state = self.state_2d()
                reward = self.reward_2d()
                action = f(state)
                states.append(state)
                actions.append([action])
                rewards.append(reward)

            self.update_rk_cons(2500 * action)
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "discounted_rewards": self.discounted_reward(rewards, y)
        }


framerate = 1000

params = {
    'g': np.array([0., 10000.]),
    'origin': np.array([int(x / 2) for x in boundary]),
    'm1': 1.,
    'framerate': 59.9,
    'x1': np.array([80., -80.]),
    'v1': np.array([0., 0.]),
    'm2': 1.,
    'x2': np.array([160., -160.]),
    'v2': np.array([0., 0.]),
    't': 2500,
}


class DQ_learning():
    def __init__(self, env_params):
        self.params = env_params
        self.env = double_pendulum(env_params)
        self.net = dense_net(3, 32, lrelu_fd, sq_diff_fd, False, 0.001)
        self.net.add_layer(32, lrelu_fd, 0.001)
        self.net.add_layer(1, linear_fd, 0.001)
        self.random_action_prob = 0.9
        self.discount = 0

    def next_action(self, state):
        if self.random_action_prob < np.random.uniform(low=0.0, high=1.0):
            pos = self.net.activate(np.append(state, 1.0))
            neg = self.net.activate(np.append(state, -1.0))
            return -np.sign(neg - pos)[0]
        return random.choice([-1.0, 1.0])

    def train_generation(self):
        traj = self.env.gen_trajectory_2c(1000, 1, f=self.next_action, y=self.discount)
        inputs = np.append(np.array(traj['states']), np.array(traj['actions']), axis=1)
        print("fitness:",self.net.update(inputs, traj['discounted_rewards'], a=-0.0000005))
        self.env.reset()
        if self.discount < 0.9:
            self.discount += 0.002
        if self.random_action_prob > 0:
            self.random_action_prob -= 0.002
        return sum(traj['rewards']), self.discount, self.random_action_prob


if __name__ == "__main__":
    dq = DQ_learning(params)
    rewards = []
    for i in range(1):
        rewards.append(dq.train_generation())
        print(rewards[-1])
        print(f'generation {i}')
        if i % 250 == 0:
            pickle.dump([rewards, dq.net], open(f"generation {i}.p", "wb"))

    pygame.init()
    pygame.display.set_caption('Double Pendulum')
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(boundary)

    dp = double_pendulum(params)


    dq.random_action_prob = 0

    steps = 0
    seconds = 0

    done = False

    timestep = 1.0 / framerate
    t = 0
    while not done:
        if steps % framerate == 0:
            # print('energy at {} : {}'.format(seconds, dp.energy()))
            # print('r1 at {} : {}'.format(seconds, np.linalg.norm(dp.p1.pos)))
            seconds += 1
        steps += 1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            t = -2000
        if keys[pygame.K_RIGHT]:
            t = 2000
        #t = -2500 * dq.next_action(dp.state_2d())
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # If user clicked close
                done = True

        screen.fill(BLACK)

        pygame.draw.line(screen, RED, dp.params['origin'], dp.p1.ppos())
        pygame.draw.line(screen, RED, dp.p1.ppos(), dp.p2.ppos())
        pygame.draw.circle(screen, WHITE, dp.p1.ppos(), dp.p1.rad())
        pygame.draw.circle(screen, WHITE, dp.p2.ppos(), dp.p2.rad())
        arrow_point = dp.p1.ppos() - max(-50, min(50, t)) * np.array(
            [-dp.p1.ppos()[1] + 250, dp.p1.ppos()[0] - 250]) / dp.r1
        pygame.draw.line(screen, GREEN, dp.p1.ppos(), [int(x) for x in arrow_point])
        pygame.display.flip()

        dp.update_rk_cons(t)
        clock.tick(framerate)
