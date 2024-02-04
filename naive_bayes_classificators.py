from typing import List, Tuple, Callable, Dict
import numpy as np
from numpy import array
from matplotlib import pyplot as plt
from random import random, choice, randint
from base_classificator_class import (
    BaseClassificator,
    TrainingRequirements,
    TrainingParams,
)
from optimisers import NotNeededOptimiser
import timeit


class NormalDistribution:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
        self.s = np.sqrt(variance)
        self.sqr2pi = np.sqrt(2) * np.pi

    def get_propability(self, x):
        value = np.exp(-((x - self.mean) ** 2) / (2 * self.variance))
        return value / (self.sqr2pi * self.s)


class NormalDistributionCalculator:
    def __init__(self):
        pass

    def calculate_propability_functions_for_items(self, X) -> List[NormalDistribution]:
        nr_of_values = len(X)
        nr_of_params = len(X[0])
        propability_functions_for_params = []
        for ip in range(nr_of_params):
            x_values = [X[ix][ip] for ix in range(nr_of_values)]
            var = np.var(x_values)
            mean = np.mean(x_values)
            func_class = NormalDistribution(mean, var)
            propability_functions_for_params.append(func_class.get_propability)
        return propability_functions_for_params


class NaiveBayesClassifier(BaseClassificator):
    def __init__(self, distribution_calc: NormalDistributionCalculator):
        super().__init__()
        self.dc = distribution_calc

    def train_on_data(self, X: List[array], Y: List[int], tr: TrainingParams):
        training_note = {}
        t1 = timeit.default_timer()
        self.Y_prob_functions = self._calculate_distributions_for_data(X, Y)

        training_note["time_training"] = timeit.default_timer() - t1
        self.training_stats.append(training_note)
        return training_note

    def _calculate_distributions_for_data(self, X: List[array], Y: list) -> dict:
        Y_X_dict = self._split_data_to_classes(X, Y)
        Y_probabilities_functions = {}
        for key, X_k in Y_X_dict.items():
            prob_functions = self.dc.calculate_propability_functions_for_items(X_k)
            Y_probabilities_functions[key] = prob_functions
        return Y_probabilities_functions

    def _split_data_to_classes(self, X: List[array], Y: list) -> dict:
        Y_objects = [Y[0]]
        Y_classes = {Y[0]: [X[0]]}
        for i in range(len(Y)):
            if Y[i] not in Y_objects:
                Y_objects.append(Y[i])
                Y_classes[Y[i]] = [X[i]]
            else:
                list_x = Y_classes[Y[i]]
                list_x.append(X[i])
                Y_classes[Y[i]] = list_x
        return Y_classes

    def get_class_scores(self, x: array):
        scores = {}
        for y, functions in self.Y_prob_functions.items():
            propability = 1
            for i in range(len(x)):
                propability = propability * functions[i](x[i])
            scores[y] = propability
        return scores

    def classify_sample(self, x: array):
        scores = self.get_class_scores(x)

        best_score = None
        best_class = None
        for key, val in scores.items():
            if best_score is None:
                best_score = val
                best_class = key
            elif best_score < val:
                best_score = val
                best_class = key
        return best_class

    def classify_sample_list(self, X: List[array]):
        Y_pred = [self.predict_class(x) for x in X]
        return Y_pred
