import numpy as np
from typing import Callable, List, Dict, Tuple
from numpy import array
from optimisers import BaseOptimiser
from random import choice
import timeit


class TrainingParams:
    def __init__(
        self,
        max_iterations: int = None,
        epsilon: float = None,
        optimiser: BaseOptimiser = None,
    ):
        self.max_iter = max_iterations
        self.eps = epsilon
        self.opt = optimiser


class TrainingRequirements:
    def __init__(self):
        self.max_iter = False
        self.eps = False
        self.opt = False


class BaseClassificator:
    def __init__(self) -> None:
        self.classes = []
        self.training_stats = []

    def get_training_requirements(self) -> TrainingRequirements:
        tr = TrainingRequirements()
        return tr

    def classify_sample(self, x: array):
        return choice(self.classes)

    def classify_sample_list(self, X: List[array]) -> List[int]:
        return [self.classify_sample(x) for x in X]

    def train_on_data(self, X: List[array], Y: List[int], tp: TrainingParams) -> dict:
        training_note = {}
        t1 = timeit.default_timer()
        for y in Y:
            if y not in self.classes:
                self.classes.append(y)

        training_note["time_training"] = timeit.default_timer() - t1
        self.training_stats.append(training_note)
        return training_note


if __name__ == "__main__":
    training_data_X = [i for i in range(10)]
    training_data_Y = [i for i in range(10)]
    optimiser = BaseOptimiser(1, 1)
    base_classifier = BaseClassificator()
    training_note = base_classifier.train_on_data(
        training_data_X, training_data_Y, optimiser
    )
    print(f"Training note: {training_note}")
