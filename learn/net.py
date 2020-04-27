import numpy as np
from learn.diff_functions import *
class dense_layer():
    def __init__(self, input_size, output_size, activation, initvar=0.2):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.initvar = initvar
        self.reset(initvar)

    def activate(self, input):
        self.input[1:] = input
        self.s = np.dot(self.weights, self.input)
        self.output = self.activation.f(self.s)
        return self.output

    def mutate(self, rate):
        self.weights += np.random.normal(0, rate,
                                         [self.output_size, self.input_size + 1]
                                         )

    def reset(self, var=0.2):
        self.weights = np.random.normal(0, var, [self.output_size, self.input_size + 1])
        self.input = np.ones(self.input_size + 1)
        self.output = np.zeros(self.output_size)
        self.derivative_buffer = 0 * self.weights
        self.derivative_buffer = 0 * self.weights


class dense_net():
    def __init__(self,
                 input_size,
                 output_size,
                 activation,
                 fitness,
                 recursive=False,
                 initvar=0.01
                 ):
        self.input_size = input_size
        self.fitness = fitness
        self.init_var = initvar
        self.layers = []
        self.recursive = recursive
        if self.recursive:
            self.input_with_state = []
        self.add_layer(output_size, activation)

    def add_layer(self, output_size, activation, initvar=0.2):
        if len(self.layers) == 0:
            self.layers.append(
                dense_layer(self.input_size, output_size, activation,
                            initvar))
        else:
            if self.recursive:
                self.layers.append(
                    dense_layer(self.layers[-1].output_size, output_size, activation,
                                initvar))
                self.layers[0] = dense_layer(
                    self.input_size + self.layers[-1].input_size,
                    self.layers[0].output_size,
                    self.layers[0].activation,
                    self.layers[0].initvar)
                self.input_with_state = np.zeros(
                    self.input_size + self.layers[-1].input_size)
            else:
                self.layers.append(
                    dense_layer(self.layers[-1].output_size, output_size, activation,
                                initvar))

    def activate(self, input):
        if self.recursive:
            self.input_with_state[:self.input_size] = input
            output = self.input_with_state
            for layer in self.layers:
                output = layer.activate(output)
            self.input_with_state[self.input_size:] = self.layers[-1].input[1:]
            return output
        else:
            output = input
            for layer in self.layers:
                output = layer.activate(output)
            return output

    def mutate(self, rate):
        if type(rate) == float:
            for layer in self.layers:
                layer.mutate(rate)
        else:
            assert (len(self.layers) == len(rate))
            for i in range(len(self.layers)):
                self.layers[i].mutate(rate[i])

    def reset(self, rate):
        if type(rate) == float:
            for layer in self.layers:
                layer.reset(rate)
        else:
            assert (len(self.layers) == len(rate))
            for i in range(len(self.layers)):
                self.layers[i].reset(rate[i])

    def _fitness(self, y0):
        # assumes that x has already been passed through
        y = self.layers[-1].output
        return self.fitness.f(y0, y)

    def _calculate_derivatives(self, y0):
        # assumes that x has already been passed through
        dldx = self.fitness.df(y0, self.layers[-1].output)
        for l in self.layers[::-1]:
            dlds = np.dot(l.activation.df(l.s), dldx)
            l.derivative = np.outer(dlds, l.input)
            dldx = np.dot(dlds, l.weights)[1:]

    def _update(self, a):
        # assumes derivatives have already been calculated
        for layer in self.layers:
            layer.weights += a * layer.derivative

    def update(self, vx, vy, a=-0.001):
        fitness = 0
        for x, y in zip(vx, vy):
            self.activate(x)
            fitness += self._fitness(y)
            self._calculate_derivatives(y)
            self._update(a)
        return fitness

    def update_trajectory(self, trajectory, discount=1.0, a=-0.001):
        vx = trajectory['state']
        vy = trajectory['action']
        rewards = trajectory['reward']
        drewards = []
        for reward in rewards:
            if len(drewards) == 0:
                drewards.append(reward)
            else:
                drewards.append(discount * drewards[-1] + reward)
        drewards = drewards[::-1]
        for x, y, z in zip(vx, vy, drewards):
            self.activate(x)
            self._calculate_derivatives(y)
            self._update(a * z)


def simple_test(recursive):
    net = dense_net(3, 10, sigmoid_fd, sq_diff_fd, recursive)
    net.add_layer(2, sigmoid_fd)
    net.add_layer(2, sigmoid_fd)
    print(net.activate(np.array([1, 1, 1])))
    print(net.activate(np.array([1, 1, 1])))
    print(net.activate(np.array([1, 1, 1])))
    net.mutate(0.01)
    print(net.activate(np.array([1, 1, 1])))
    print(net.activate(np.array([1, 1, 1])))
    print(net._fitness([0, 0]))
    net._calculate_derivatives([0, 0])


def run_stress(recursive):
    net = dense_net(2, 128, sigmoid_fd, sq_diff_fd, recursive)
    net.add_layer(128, sigmoid_fd)
    net.add_layer(128, sigmoid_fd)
    net.add_layer(128, sigmoid_fd)
    net.add_layer(128, sigmoid_fd)
    net.add_layer(2, sigmoid_fd)
    input = np.array([1, 1, 1])
    for i in range(1000):
        net.activate(np.array(input))


def training_stress():
    net = dense_net(1, 128, lrelu_fd, sq_diff_fd, False)
    net.add_layer(128, lrelu_fd)
    net.add_layer(1, linear_fd)
    for i in range(1000000):
        x = np.random.uniform(low=-25.0, high=25.0)

        input = np.array([x])
        net.activate(input)
        if i % 500 == 0:
            print(net._fitness(input ** 2))
        net._calculate_derivatives(input ** 2)
        net._update(-0.00001)


def time_test(recursive):
    import cProfile
    cProfile.run("run_stress({})".format(recursive))


def train_time_test():
    import cProfile
    cProfile.run("training_stress()")


if __name__ == "__main__":
    training_stress()
    simple_test(True)
    simple_test(False)
    time_test(True)
    time_test(False)
    train_time_test()
