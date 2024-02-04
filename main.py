from naive_bayes_classificators import (
    NaiveBayesClassifier,
    NormalDistributionCalculator,
)
from performance_testing_classes import BinaryClassifficatorPerformanceAnaliser
from base_classificator_class import BaseClassificator, TrainingParams
from typing import List, Tuple
from random import random
from copy import deepcopy
from numpy import array
from ucimlrepo import fetch_ucirepo
import pandas as pd
from functools import partial
from data_operator_classes import BaseDataOperator


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
    bdo = BaseDataOperator()

    Yb = bdo.binarise_data(Y, 6)
    print(bdo.get_nr_of_samples_per_class(Yb))

    tp = TrainingParams()
    nbc = NaiveBayesClassifier()
    nbc.train_on_data(X, Yb, tp)
    print(bdo.get_nr_of_samples_per_class(nbc.classify_sample_list(X)))
    nanal = BinaryClassifficatorPerformanceAnaliser(nbc)
    print(nanal.get_full_performance_analisys(X, Yb))
    bc = BaseClassificator()
