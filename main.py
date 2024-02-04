from naive_bayes_classificators import NaiveBayesClassifier
from base_classificator_class import BaseClassificator, TrainingParams
from typing import List, Tuple
from random import random


class BinaryClassifierPerformanceAnaliser:
    def __init__(
        self, X, Y, classifier: BaseClassificator, training_params: TrainingParams
    ):
        self.tp = TrainingParams
        self.classifier = classifier
        self.X = X
        self.Y = Y
        self.classes = self._get_all_possible_classes(Y)

    def _get_all_possible_classes(self, Y):
        classes = []
        for y in Y:
            if y not in classes:
                classes.append(y)
        return classes

    def get_n_of_classes(self, nr_to_get_per_class: dict, X, Y) -> Tuple[list, list]:
        new_y = []
        new_x = []
        # Your list of indexes
        index_list = [*range(len(Y))]
        random.shuffle(index_list)
        for i in index_list:
            nr_to_get = nr_to_get_per_class[Y[i]]
            if nr_to_get > 0:
                new_y.append(Y[i])
                new_x.append(X[i])
                nr_to_get_per_class[Y[i]] = nr_to_get - 1
        return new_x, new_y

    def create_confusion_dict(self, Y_data, Y_predicted) -> dict:
        tp_n = 0
        fp_n = 0
        tn_n = 0
        fn_n = 0
        data_size = len(Y_data)
        for i in range(data_size):
            yo = Y_data[i]
            yp = Y_data[i]
            if yo == yp:
                if yp == 1:
                    tp_n += 1
                else:
                    tn_n += 1
            else:
                if yp == 1:
                    fp_n += 1
                else:
                    fn_n += 1
        confusion_dict = {}
        confusion_dict["tp"] = tp_n / data_size
        confusion_dict["fp"] = fp_n / data_size
        confusion_dict["tn"] = tn_n / data_size
        confusion_dict["fn"] = fn_n / data_size
        return confusion_dict

    def _calculate_error(self, conf_dict: dict) -> float:
        error = conf_dict["fp"] + conf_dict["fn"]
        error = error / (
            conf_dict["fp"] + conf_dict["fn"] + conf_dict["tp"] + conf_dict["tn"]
        )
        return error

    def _calculate_accuracy(self, conf_dict: dict) -> float:
        error = conf_dict["tp"] + conf_dict["tn"]
        error = error / (
            conf_dict["fp"] + conf_dict["fn"] + conf_dict["tp"] + conf_dict["tn"]
        )
        return error

    def calculate_true_positive_false_positive_rates(
        self, conf_dict: dict
    ) -> Tuple[float, float]:
        tpr = conf_dict["tp"] / (conf_dict["tp"] + conf_dict["fn"])
        fpr = conf_dict["fp"] / (conf_dict["tn"] + conf_dict["fp"])
        return tpr, fpr

    def calculate_extended_accuracy_metrics(self, conf_dict: dict) -> dict:
        accuracy_metrics = {}
        accuracy_metrics["accuracy"] = self._calculate_accuracy(conf_dict)
        accuracy_metrics["error"] = self._calculate_error(conf_dict)
        tpr, fpr = self.calculate_true_positive_false_positive_rates(conf_dict)
        accuracy_metrics["tpr"] = tpr
        accuracy_metrics["fpr"] = fpr
