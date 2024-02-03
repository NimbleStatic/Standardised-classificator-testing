import numpy as np
from numpy import array


class BaseOptimiser:
    def __init__(self, initial_step_size, decay):
        self.decay = decay
        self.init_step_size = initial_step_size
        self.last_step_size = initial_step_size

    def calculate_descented_x(self, x0, gradient: array):
        new_ss = self.last_step_size**self.decay
        self.last_step_size = new_ss
        return x0 - new_ss * gradient


class AdamOptimiser(BaseOptimiser):
    def __init__(
        self, alpha: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps=1e-8
    ):
        self.init_alpha = alpha
        self.alpha = alpha
        self.init_b1 = beta1
        self.init_b2 = beta2
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.m = 0
        self.v = 0
        self.beta_history = []
        self.iterations = 0

    def calculate_descented_x(self, x0: array, gradient: array) -> array:
        self.iterations += 1
        m = self.b1 * self.m + (1 - self.b1) * gradient
        v = self.b2 * self.v + (1 - self.b2) * (np.power(gradient, 2))
        mh = m / (1 - self.b1)
        vh = v / (1 - self.b2)

        self.b1 = self.b1**self.iterations
        self.b2 = self.b2**self.iterations
        # self.b1 = self.init_b1**self.iterations
        # self.b2 = self.init_b2**self.iterations
        value = (self.alpha * mh) / (np.sqrt(vh) + self.eps)
        self.m = m
        self.v = v
        return x0 - value
