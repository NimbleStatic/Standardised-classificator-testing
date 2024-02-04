from naive_bayes_classificators import NaiveBayesClassifier
from base_classificator_class import BaseClassificator, TrainingParams
from typing import List, Tuple
from random import random
from copy import deepcopy
from numpy import array
from ucimlrepo import fetch_ucirepo
import pandas as pd
from functools import partial


class BinaryClassifficatorPerformanceAnaliser:
    def __init__(self, classifier: BaseClassificator):
        self.classifier = classifier

    def predict_with_classifier(self, X: List[array]):
        predicted_Y = self.classifier.classify_sample_list(X)
        return predicted_Y

    # def _get_all_possible_classes(self, Y: List[int]):
    #     classes = []
    #     for y in Y:
    #         if y not in classes:
    #             classes.append(y)
    #     return classes

    # def get_n_of_classes(self, nr_to_get_per_class: dict, X, Y) -> Tuple[list, list]:
    #     new_y = []
    #     new_x = []
    #     index_list = [*range(len(Y))]
    #     random.shuffle(index_list)
    #     for i in index_list:
    #         nr_to_get = nr_to_get_per_class[Y[i]]
    #         if nr_to_get > 0:
    #             new_y.append(Y[i])
    #             new_x.append(X[i])
    #             nr_to_get_per_class[Y[i]] = nr_to_get - 1
    #     return new_x, new_y

    def create_confusion_dict(self, Y_data, Y_predicted) -> dict:
        tp_n = 0
        fp_n = 0
        tn_n = 0
        fn_n = 0
        data_size = len(Y_data)
        for i in range(data_size):
            yo = Y_data[i]
            yp = Y_predicted[i]
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
        confusion_dict["tp"] = tp_n
        confusion_dict["fp"] = fp_n
        confusion_dict["tn"] = tn_n
        confusion_dict["fn"] = fn_n
        return confusion_dict

    def calculate_error(self, conf_dict: dict) -> float:
        error = conf_dict["fp"] + conf_dict["fn"]
        error = error / (
            conf_dict["fp"] + conf_dict["fn"] + conf_dict["tp"] + conf_dict["tn"]
        )
        return error

    def calculate_accuracy(self, conf_dict: dict) -> float:
        accuracy = conf_dict["tp"] + conf_dict["tn"]
        divider = conf_dict["fp"] + conf_dict["fn"] + conf_dict["tp"] + conf_dict["tn"]
        # if divider == 0:
        #     return 0
        accuracy = accuracy / divider
        return accuracy

    def calculate_true_positive_rate(self, conf_dict: dict) -> float:
        tpr = conf_dict["tp"] / (conf_dict["tp"] + conf_dict["fn"])
        return tpr

    def calculate_false_positive_rate(self, conf_dict: dict) -> float:
        fpr = conf_dict["fp"] / (conf_dict["tn"] + conf_dict["fp"])
        return fpr

    def calculate_f_measure(self, conf_dict: dict, tpr: float = None) -> float:
        if tpr is None:
            tpr = self.calculate_true_positive_rate(conf_dict)
        precision = conf_dict["tp"] / (conf_dict["tp"] + conf_dict["fp"])
        fmeasure = 2 * tpr * precision / (tpr + precision)
        return fmeasure

    def calculate_extended_accuracy_metrics(self, conf_dict: dict) -> dict:
        accuracy_metrics = {}
        accuracy_metrics["accuracy"] = self.calculate_accuracy(conf_dict)
        accuracy_metrics["error"] = self.calculate_error(conf_dict)
        tpr = self.calculate_true_positive_rate(conf_dict)
        accuracy_metrics["f_measure"] = self.calculate_f_measure(conf_dict, tpr)
        accuracy_metrics["tpr"] = tpr
        accuracy_metrics["fpr"] = self.calculate_false_positive_rate

        return accuracy_metrics

    def get_full_performance_analisys(self, X, Y):
        Ypred = self.predict_with_classifier(X)
        conf_dict = self.create_confusion_dict(Y, Ypred)
        accuracy_metrics = self.calculate_extended_accuracy_metrics(conf_dict)
        return accuracy_metrics


def save_data_to_csv(data_id: int = 186, path: str = "training_data.csv"):

    data_set = fetch_ucirepo(id=data_id)
    X_data = data_set.data.features
    y_data = data_set.data.targets
    df_x = pd.DataFrame(X_data)
    df_y = pd.DataFrame(y_data)
    x_name = "x_" + path
    y_name = "y_" + path
    df_x.to_csv(x_name)
    df_y.to_csv(y_name)


def get_x_y_train_data(
    path: str = "training_data.csv",
) -> Tuple[List[array], List[float]]:
    x_name = "x_" + path
    y_name = "y_" + path
    df_x = pd.read_csv(x_name)
    disc_df_y = pd.read_csv(y_name)

    X_tries = df_x.to_numpy()
    Y_tries = disc_df_y.to_numpy()
    x_tries_form = []
    for X in X_tries:
        x_tries_form.append(X[1:])
    Y_tries_form = []
    for y in Y_tries:
        Y_tries_form.append(float(y[1]))
    return x_tries_form, Y_tries_form


if __name__ == "__main__":
    # save_data_to_csv()
    X, Y = get_x_y_train_data()
    classifier = BaseClassificator()
    tp = TrainingParams()
    bcpa = BinaryClassifierPerformanceAnaliser(X, Y, classifier, tp)
    classifier_trained = bcpa._train_classifier(X, Y, classifier)
    yp_init = bcpa.predict_with_classifier(X, classifier_trained)
    print(yp_init)
    print(
        bcpa.calculate_extended_accuracy_metrics(bcpa.create_confusion_dict(Y, yp_init))
    )
