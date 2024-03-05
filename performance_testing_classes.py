from base_classificator_class import BaseClassificator, TrainingParams
from typing import List, Tuple, Dict
from random import random
from copy import deepcopy
from numpy import array


class BinaryClassifficatorPerformanceAnaliser:
    def __init__(self):
        pass

    def create_confusion_dict(self, Y_data: List[int], Y_predicted: List[int]) -> dict:
        # positive is 1 negative is everything else
        tp_n = 0
        fp_n = 0
        tn_n = 0
        fn_n = 0
        for i in range(len(Y_data)):
            yo = Y_data[i]
            yp = Y_predicted[i]
            if float(yp) == float(yo) == 1:
                tp_n += 1
            elif float(yp) == float(yo):
                tn_n += 1
            elif float(yp) == 1:
                fp_n += 1
            else:
                fn_n += 1

        confusion_dict = {"tp": tp_n, "fp": fp_n, "tn": tn_n, "fn": fn_n}
        return confusion_dict

    def calculate_error(self, conf_dict: dict) -> float:
        divider = conf_dict["fp"] + conf_dict["fn"] + conf_dict["tp"] + conf_dict["tn"]
        if divider == 0:
            return 0
        error = (conf_dict["fp"] + conf_dict["fn"]) / divider
        return error

    def calculate_accuracy(self, conf_dict: dict) -> float:
        divider = conf_dict["fp"] + conf_dict["fn"] + conf_dict["tp"] + conf_dict["tn"]
        if divider == 0:
            return 0
        accuracy = (conf_dict["tp"] + conf_dict["tn"]) / divider
        return accuracy

    def calculate_true_positive_rate(self, conf_dict: dict) -> float:
        divider = conf_dict["tp"] + conf_dict["fn"]
        if divider == 0:
            return 0
        tpr = conf_dict["tp"] / divider
        return tpr

    def calculate_false_positive_rate(self, conf_dict: dict) -> float:
        divider = conf_dict["tn"] + conf_dict["fp"]
        if divider == 0:
            return 0
        fpr = conf_dict["fp"] / divider
        return fpr

    def calculate_f_measure(self, conf_dict: dict, tpr: float = None) -> float:
        if tpr is None:
            tpr = self.calculate_true_positive_rate(conf_dict)
        divider = conf_dict["tp"] + conf_dict["fp"]
        if divider == 0:
            precision = 0
            if tpr == 0:
                return 0
        else:
            precision = conf_dict["tp"] / divider

        fmeasure = 2 * tpr * precision / (tpr + precision)
        return fmeasure

    def calculate_extended_accuracy_metrics(self, conf_dict: dict) -> dict:
        accuracy_metrics = {}
        accuracy_metrics["accuracy"] = self.calculate_accuracy(conf_dict)
        accuracy_metrics["error"] = self.calculate_error(conf_dict)
        tpr = self.calculate_true_positive_rate(conf_dict)
        accuracy_metrics["f_measure"] = self.calculate_f_measure(conf_dict, tpr)
        accuracy_metrics["tpr"] = tpr
        accuracy_metrics["fpr"] = self.calculate_false_positive_rate(conf_dict)

        return accuracy_metrics

    def get_full_performance_analisys(self, X, Y, Y_predicted) -> Dict[str, float]:
        conf_dict = self.create_confusion_dict(Y, Y_predicted)
        accuracy_metrics = self.calculate_extended_accuracy_metrics(conf_dict)
        return accuracy_metrics
