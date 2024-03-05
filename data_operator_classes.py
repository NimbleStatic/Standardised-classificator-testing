import numpy as np
from typing import Callable, List, Dict, Tuple
from numpy import array
import random


class ClassificationDataConverter:
    def __init__(self):
        self.conversion_dict_to_key = {}
        self.conversion_range = (None, None)

    def _create_reverse_dict(self, dict_to_reverse):
        pass

    def convert_data_classes_to_int(self, Y: List):
        pass


class BaseDataOperator:
    def __init__(self):
        pass

    def select_data_from_Y(
        self, X: List[any], Y: List[any], classes_to_retain: List[any]
    ):
        new_x = []
        new_y = []
        for i in range(len(Y)):
            if Y[i] in classes_to_retain:
                new_x.append(X[i])
                new_y.append(Y[i])
        return new_x, new_y

    def binarise_data(self, Y: List[float], binarisation_level: float):
        new_y = []
        for y in Y:
            if y >= binarisation_level:
                new_y.append(1)
            else:
                new_y.append(-1)
        return new_y

    def choose_data_randomly_per_class(
        self, nr_to_get_per_class: Dict[any, int], X: List[any], Y: List[any]
    ):
        new_y = []
        new_x = []
        index_list = [*range(len(Y))]
        random.shuffle(index_list)
        for i in index_list:
            nr_to_get = nr_to_get_per_class[Y[i]]
            if nr_to_get > 0:
                new_y.append(Y[i])
                new_x.append(X[i])
                nr_to_get_per_class[Y[i]] = nr_to_get - 1
        return new_x, new_y

    def get_possible_classes(self, Y: List[any]) -> List:
        classes = []
        for y in Y:
            if y not in classes:
                classes.append(y)
        return classes

    def get_nr_of_samples_per_class(self, Y: List[any]) -> Dict[any, int]:
        count_per_class = {}
        for y in Y:
            if y in count_per_class:
                count_per_class[y] += 1
            else:
                count_per_class[y] = 1
        count_per_class = dict(sorted(count_per_class.items()))
        return count_per_class


class PermanentDataOperations:
    def __init__(self, X, Y):
        # The class instance is specific to the data inputed in the init, dont use with different data sets
        self.X = X
        self.Y = Y
        self._calculate_normalisation_parameter_limits()

    def _calculate_normalisation_parameter_limits(self):
        self.min_limit = list(self.X[0])
        self.max_limit = list(self.X[0])

        for x in self.X[1:]:
            for i, value in enumerate(x):
                if value < self.min_limit[i]:
                    self.min_limit[i] = value
                if value > self.max_limit[i]:
                    self.max_limit[i] = value

        self.min_limit = array(self.min_limit)
        self.max_limit = array(self.max_limit)
        self.val_amplitude = (self.max_limit) - (self.min_limit)

    def normalise_sample(self, x: array):
        x_capped = np.clip(x, self.min_limit, self.max_limit)
        x_normalised = (x_capped - self.min_limit) / self.val_amplitude
        return x_normalised

    def normalise_sample_list(self, X: List[array]):
        return [self.normalise_sample(x) for x in X]
