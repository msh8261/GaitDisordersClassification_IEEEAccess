import os

import matplotlib.pyplot as plt
import numpy as np

import config.config_train as config
# from prepare_dataset.best_features_selection.FSM import FSM
from prepare_dataset.tools.decorators import logger
# from prepare_dataset.tools.decorators import timeit
# from src.dataset import GaitData
from src.load import LoadData

# import pandas as pd


@logger
def show_resutls(result, method_name):
    print(f"=================== results of {method_name} ===================")
    print(f"best featurs: {result}")
    if result is not None:
        print(f"number of best features: {len(result)}")


@logger
def get_data(X_path, y_path, mode):
    ld = LoadData(X_path, y_path, config.params["num_augmentation"], False)
    X = ld.get_X()
    y = ld.get_y()
    return X, y


@logger
def clean_arr_save_to_file(arr, no_best_features):
    if len(arr) > no_best_features:
        arr = arr[0:no_best_features]
    if not all(isinstance(x, int) for x in arr):
        arr = [(val.replace("x", "")) for val in arr]
    arr = [str(item) for item in arr]
    arr = ["%s" % item for item in arr]
    arr = ",".join(arr)
    return arr


def plot_features_by_label_mean(X, y, no_best_features):
    lbs = ["0", "1", "2"]
    colors = ["b", "g", "m"]
    fig, ax = plt.subplots(len(lbs), 1, figsize=(6, 6))
    for j in range(len(lbs)):
        data = [X[k].mean() for k in range(len(y)) if str(y[k]) == lbs[j]]
        label = f"result of {no_best_features} best features for label {lbs[j]}"
        ax[j].plot(data, label=label, color=colors[j])
        ax[j].legend(loc="upper right")
        ax[j].grid()
    plt.tight_layout()
    plt.show()


def plot_all_sample_features_mean(X, no_best_features):
    x_mean = [x.mean() for x in X]
    plt.plot(x_mean, color="m")
    plt.title(f"result of {no_best_features} best features")
    plt.grid()
    plt.show()


def plot_a_sample_features(X, y, no_best_features):
    for j in range(len(X)):
        plt.plot(X[j], color="m")
        plt.title(f"result of {no_best_features} best features, target {y[j]}")
        plt.grid()
        plt.show()


@logger
def show_features_in_sequences(
    X_path, y_path, mode, no_best_features, mode_of_show, no_seq_show=None
):
    X_, y_ = get_data(X_path, y_path, mode)
    sequences_ = no_seq_show
    if no_seq_show is None:
        sequences_ = X_.shape[1]
    for i in range(sequences_):
        # array of features in sequences
        X = np.array([x[i] for x in X_])
        y = np.array([int(val) for val in y_])
        # print(X.shape)
        if mode_of_show == "by_lable":
            plot_features_by_label_mean(X, y, no_best_features)
        elif mode_of_show == "all_samples":
            plot_all_sample_features_mean(X, no_best_features)
        elif mode_of_show == "a_sample":
            plot_a_sample_features(X, y, no_best_features)
        else:
            assert "INFO: Please specify the mode of show"


if __name__ == "__main__":
    original_total_features = 70
    no_best_features = 10

    show_best_features = True
    no_seq_show = 2

    # train or test
    mode_name = "train"
    list_method_name = [
        "ANOVA",
        "chi2",
        "kruskal",
        "cmim",
        "disr",
        "mifs",
        "ReliefF",
        "SURF",
        "SURFstar",
        "MultiSURF",
    ]
    method_name = list_method_name[0]

    modes = ["detection", "tracking"]

    for mode in modes:
        if show_best_features:
            dir_to_final_features_file = f"data/{mode}/best_feats/final_{no_best_features}_best_feats_selections_from_{original_total_features}"
            X_path = os.path.join(
                dir_to_final_features_file,
                f"X{mode_name}_{method_name}_{no_best_features}_selected_best_feats.File",
            )
            y_path = os.path.join(dir_to_final_features_file, f"y{mode_name}.File")
        else:
            dataset_path = f"./data/{mode}/{original_total_features}_zero_padding_no"
            X_path = os.path.join(dataset_path, f"X{mode_name}.File")
            y_path = os.path.join(dataset_path, f"y{mode_name}.File")

        # show_features_in_sequences(X_path, y_path, mode_name, no_best_features, 'all_samples', no_seq_show)
        show_features_in_sequences(
            X_path, y_path, mode_name, no_best_features, "a_sample", no_seq_show
        )
        # show_features_in_sequences(X_path, y_path, mode_name, no_best_features, 'by_lable', no_seq_show)
