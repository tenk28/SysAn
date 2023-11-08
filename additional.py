import enum

import numpy as np
from scipy.sparse.linalg import cg


class Polynom(enum.Enum):
    CHEBYSHEV = 1
    LEGANDRE = 2
    LAGERR = 3


class Weight(enum.Enum):
    NORMED = 1
    MIN_MAX = 2


class Lambda(enum.Enum):
    SINGLE_SET = 1
    TRIPLE_SET = 2


def chebyshev_bias_value(x, n):
    if n == 0:
        return 1
    elif n == 1:
        return 2 * x - 1
    elif n == 2:
        return 8 * x**2 - 8 * x + 1
    elif n == 3:
        return 32 * x**3 - 48 * x**2 + 18 * x - 1
    elif n == 4:
        return 128 * x**4 - 256 * x**3 + 160 * x**2 - 32 * x + 1


def chebyshev_bias_polynomial(n):
    if n == 0:
        return np.array([1])
    elif n == 1:
        return np.array([2, -1])
    elif n == 2:
        return np.array([8, -8, 1])
    elif n == 3:
        return np.array([32, -48, 18, -1])
    elif n == 4:
        return np.array([128, -256, 160, -32, 1])


def legandre_value(x, n):
    if n == 0:
        return 1
    elif n == 1:
        return 2 * x - 1
    elif n == 2:
        return -6 * x**2 + 6 * x - 1
    elif n == 3:
        return 20 * x**3 - 30 * x**2 + 12 * x - 1
    elif n == 4:
        return -70 * x**4 + 140 * x**3 - 90 * x**2 + 20 * x - 1


def legandre_polynomial(n):
    if n == 0:
        return np.array([1])
    elif n == 1:
        return np.array([2, -1])
    elif n == 2:
        return np.array([-6, 6, -1])
    elif n == 3:
        return np.array([20, -30, 12, -1])
    elif n == 4:
        return np.array([-70, 140, -90, 20, -1])


def lagerr_value(x, n):
    if n == 0:
        return 1
    elif n == 1:
        return -x + 1
    elif n == 2:
        return x**2 - 4 * x + 2
    elif n == 3:
        return -(x**3) + 9 * x**2 - 18 * x + 6
    elif n == 4:
        return x**4 - 16 * x**3 + 72 * x**2 - 96 * x + 24


def lagerr_polynomial(n):
    if n == 0:
        return np.array([1])
    elif n == 1:
        return np.array([-1, 1])
    elif n == 2:
        return np.array([1, -4, 2])
    elif n == 3:
        return np.array([-1, 9, -18, 6])
    elif n == 4:
        return np.array([1, -16, 72, -96, 24])


def functional(A, x, b):
    return np.max(np.abs(A @ x - b))


def functional_grad(A, x, b):
    ind = np.argmax(np.abs(A @ x - b))
    max_val = (A @ x - b)[ind]
    return (A[ind] if max_val >= 0 else -A[ind]).reshape(-1, 1)


def gradient(plug, A, b, max_iteration=100_000, eps=1e-6):
    learning_rate = 0.001
    x = np.zeros((A.shape[1], 1))
    i = 0
    while i < max_iteration and functional(A, x, b) >= eps:
        grads = functional_grad(A, x, b)
        x = x - learning_rate * grads
        i += 1
    return x


def adagrad(plug, A, b, max_iteration=100_000, eps=1e-6):
    learning_rate = 0.01
    x = np.zeros((A.shape[1], 1))
    i = 0
    g_t = functional_grad(A, x, b) ** 2
    while i < max_iteration and functional(A, x, b) >= eps:
        x = x - (learning_rate * functional_grad(A, x, b) / np.sqrt(g_t + eps))
        g_t += functional_grad(A, x, b) ** 2
        i += 1
    return x


def conjugate_gradient_method(plug, A, b, eps=1e-6):
    # return np.linalg.lstsq(A, b)[0]
    return np.matrix(cg(A.T @ A, A.T @ b, tol=eps)[0]).reshape(-1, 1)
