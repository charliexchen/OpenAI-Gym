import numpy as np


class function_with_derivative:
    def __init__(self, f, df, extra_args=None):
        self.f = f
        self.df = df


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return np.diag(np.exp(-x) / ((1 + np.exp(-x)) ** 2))


sigmoid_fd = function_with_derivative(sigmoid, d_sigmoid)


def lrelu(x):
    return np.maximum(x, 0.0)


def d_lrelu(x):
    return np.diag(np.where(x > 0, 1.0, 0.0))


lrelu_fd = function_with_derivative(lrelu, d_lrelu)


def tanh(x, scale=1.0):
    return scale * np.tanh(x)


def dtanh(x, scale=1.0):
    return np.diag(scale / (np.cosh(x) ** 2))


tanh_fd = function_with_derivative(tanh, dtanh)


def square_diff(y0, y):
    return np.dot(y0 - y, y0 - y)


def d_square_diff(y0, y):
    return -2 * (y0 - y)


sq_diff_fd = function_with_derivative(square_diff, d_square_diff)


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)


def d_softmax(x):
    exp = np.exp(x)
    denum = np.sum(exp)
    diag = np.diag(exp / denum)

    off_diag = np.array(np.matrix([
        np.array([-(exp[i]*exp[j])/ denum ** 2
            for j in range(len(x))
        ]) for i in range(len(x))
    ]))
    return diag+off_diag


softmax_fd = function_with_derivative(softmax, d_softmax)


def softmax_i(x, i):
    exp = np.exp(x)
    return exp[i] / np.sum(exp)


def d_softmax_i(x, i):
    exp = np.exp(x)
    return -(exp[i] / np.sum(exp)) ** 2


softmax_i_fd = function_with_derivative(softmax_i, d_softmax_i)


def log(x, ind):
    return np.log(x)


def d_log(x):
    return 1 / x


log_fd = function_with_derivative(log, d_log)


def log_pol(ind, x):
    return np.log(x[ind])


def d_log_pol(ind, x):
    return np.array([min(1 / x[i], 1000) if i == ind else 0 for i in range(len(x))])


log_pol_fd = function_with_derivative(log_pol, d_log_pol)


def cross_entropy(p, p0):
    return -np.dot(p0, np.log(p))


def d_cross_entropy(p, p0):
    return -np.dot(p0, 1 / p)


softmax_i_fd = function_with_derivative(cross_entropy, d_cross_entropy)


def linear(x):
    return x


def d_linear(x):
    return 1


linear_fd = function_with_derivative(linear, d_linear)

def huber(y0, y):
    sq =  np.dot(y0 - y, y0 - y)
    if sq < 1:
        return sq/2
    else:
        return np.sqrt(sq)-0.5

def d_huber(y0, y):
    sq = np.dot(y0 - y, y0 - y)

    if sq < 1:
        return  -(y0 - y)
    else:
        return  -(y0 - y)/np.linalg.norm(y0 - y)


huber_fd = function_with_derivative(huber, d_huber)
