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

    def get_propability(self, x: float):
        value = np.exp(-((x - self.mean) ** 2) / (2 * self.variance))
        return value / (self.sqr2pi * self.s)


class NormalDistributionCalculator:
    def __init__(self):
        pass

    def calculate_propability_functions_for_items(self, X) -> List[callable]:
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


class DiscreteDistribution:
    def __init__(self, calculated_propabilities: Dict[Tuple[float, float], float]):
        self.propability_values = calculated_propabilities

    def get_propability(self, x: float):
        for range, value in self.propability_values.items():
            if range[0] <= x < range[1]:
                return value
        return 0


class DiscreteDistributionCalculator(NormalDistributionCalculator):
    def __init__(self, nr_of_ranges: int):
        self.range_nr = nr_of_ranges

    def calculate_propability_functions_for_items(self, X) -> List[callable]:
        nr_of_values = len(X)
        nr_of_params = len(X[0])
        propability_functions_for_params = []
        for ip in range(nr_of_params):
            x_values = [X[ix][ip] for ix in range(nr_of_values)]
            range_propability_dict = {}
            ranges = self._calculate_ranges_for_classification(x_values)
            for val_range in ranges:
                nr_in_range = 0
                for x in x_values:
                    if val_range[0] <= x < val_range[1]:
                        nr_in_range += 1
                range_propability_dict[val_range] = nr_in_range / nr_of_values
            func_class = DiscreteDistribution(range_propability_dict)
            propability_functions_for_params.append(func_class.get_propability)
        return propability_functions_for_params

    def _calculate_ranges_for_classification(self, x_values: List[float]):
        min_x = min(x_values)
        max_x = max(x_values)
        range_size = (max_x - min_x) / self.range_nr
        ranges = [
            ((min_x + i * range_size), (min_x + (i + 1) * range_size))
            for i in range(self.range_nr)
        ]
        # for i in range(self.range_nr):
        #     size_tuple = ((min_x + i * range_size), (min_x + (i + 1) * range_size))
        return ranges


class NaiveBayesClassifier(BaseClassificator):
    def __init__(self, distribution_calc: NormalDistributionCalculator = None):
        super().__init__()
        if distribution_calc is None:
            distribution_calc = NormalDistributionCalculator()
        self.dc = distribution_calc

    def train_on_data(
        self,
        X: List[array],
        Y: List[int],
        tp: TrainingParams,
        # atribute_value_filter: List[float] = None,
    ):
        training_note = {}
        # if atribute_value_filter is None:
        #     atribute_value_filter = [1 for _ in X[0]]
        # self.atrbute_filter = atribute_value_filter
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
        Y_pred = [self.classify_sample(x) for x in X]
        return Y_pred


class SemiNaiveBayesClassifier(NaiveBayesClassifier):
    def __init__(
        self,
        distribution_calc: NormalDistributionCalculator = None,
        dampening_power: float = 0.2,
    ):
        self.dampening_power = dampening_power
        super().__init__(distribution_calc)

    def get_class_scores(self, x: array):
        scores = {}
        for y, functions in self.Y_prob_functions.items():
            propability = 1
            for i in range(len(x)):
                value = functions[i](x[i])
                dampened_value = (value) ** (
                    1 + (self.dampening_power * self.dampening[i])
                )
                propability = propability * dampened_value
            scores[y] = propability
        return scores

    def train_on_data(self, X: List, Y: List[int], tp: TrainingParams):
        self.dampening = self.calculate_mean_abs_correlation(X)
        return super().train_on_data(X, Y, tp)

    def calculate_correlation_of_atributes(self, X1: List[float], X2: List[float]):
        meanx1 = np.average(X1)
        meanx2 = np.average(X2)
        covariance = 0
        standard_devaition_x1 = 0
        standard_devaition_x2 = 0
        for i in range(len(X1)):
            x1 = X1[i]
            x2 = X2[i]
            covariance += (x1 - meanx1) * (x2 - meanx2)
            standard_devaition_x1 += (x1 - meanx1) ** 2
            standard_devaition_x2 += (x2 - meanx2) ** 2
        return covariance / ((standard_devaition_x1 * standard_devaition_x2) ** 0.5)

    def separate_attributes(self, X: List[array]):
        nr_of_attr = len(X[0])
        separated = []
        for i in range(nr_of_attr):
            X_s = [x[i] for x in X]
            separated.append(X_s)
        return separated

    def calculate_mean_abs_correlation(self, X: List[array]):

        attributes = self.separate_attributes(X)
        mean_attr_correlations = []
        for ix1, X1 in enumerate(attributes):
            correlations = []
            for ix2, X2 in enumerate(attributes):
                if ix1 != ix2:
                    correlations.append(
                        abs(self.calculate_correlation_of_atributes(X1, X2))
                    )
            mean_attr_correlations.append(np.mean(correlations))
        return mean_attr_correlations
