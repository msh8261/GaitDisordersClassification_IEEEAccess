# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103, e1101
import os

# import ast
# import glob
# import pandas as pd
# import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
# import numpy as np
import torch
# import torch.nn as nn
import torch.nn.functional as F
# import sys
# import yaml
from matplotlib import cycler
# from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
#                              accuracy_score, average_precision_score,
#                              classification_report, confusion_matrix, f1_score,
#                              precision_recall_curve)
# from sklearn.metrics import precision_recall_fscore_support as score
# from sklearn import metrics
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
# from sklearn.preprocessing import label_binarize
# from random import shuffle
from sklearn.utils import shuffle

from src.dataset import GaitData
from src.load import LoadData

# from itertools import cycle


colors = cycler(
    "color", ["#EE6666", "#3388BB", "#9988DD", "#EECC55", "#88BB44", "#FFBBBB"]
)
plt.rc(
    "axes",
    facecolor="#E6E6E6",
    edgecolor="none",
    titlesize=14,
    axisbelow=True,
    grid=True,
    prop_cycle=colors,
)
plt.rc("grid", color="w", linestyle="solid")
plt.rc("xtick", direction="out", color="gray", labelsize=14)
plt.rc("ytick", direction="out", color="gray", labelsize=14)
plt.rc("patch", edgecolor="#E6E6E6")
plt.rc("lines", linewidth=2, linestyle="-.")
plt.rcParams["figure.figsize"] = (7, 5)
matplotlib.rc("font", size=14)


# the axes attributes need to be set before the call to subplot
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc("font", weight="bold")
plt.rc("xtick.major", size=5, pad=7)
plt.rc("xtick", labelsize=12)
font = {"family": "serif", "weight": "bold", "size": 12}

plt.rc("font", **font)


# using aliases for color, linestyle and linewidth; gray, solid, thick
plt.rc("grid", c="0.5", ls="-", lw=1.5)
plt.rc("lines", lw=2, color="g")


device = torch.device("cuda" if torch.cuda.is_available() else "cup")


def metrics_in_file(labels_gt, labels, bf_method, input_size, select_kind_62, f):
    acc = accuracy_score(labels_gt, labels).round(2)
    ave_fscore = f1_score(labels_gt, labels, average="macro").round(2)
    ave_precision = precision_score(labels_gt, labels, average="macro").round(2)
    ave_recall = recall_score(labels_gt, labels, average="macro").round(2)

    if bf_method:
        f.write(
            f"{bf_method} & {input_size} & {acc} & {ave_fscore} & {ave_precision} & {ave_recall}\n"
        )
    else:
        if str(input_size) == "62":
            f.write(
                f"{input_size}_{select_kind_62} & {acc} & {ave_fscore} & {ave_precision} & {ave_recall}\n"
            )
        else:
            f.write(
                f"{input_size} & {acc} & {ave_fscore} & {ave_precision} & {ave_recall}\n"
            )


def read_test_model(
    f, path_model, model_name, test_dataset, input_size, mode, bf_method, select_kind_62
):
    labels, preds = [], []
    model_saved = torch.jit.load(path_model)
    model_saved.to(device)
    model_saved.eval()
    print("============ Test Results =============")
    print(f"model {model_name} is running for test....")
    test_dataset = shuffle(test_dataset)
    for ix in range(len(test_dataset)):
        im, label = test_dataset[ix]
        data_input = torch.autograd.Variable(torch.tensor(im[None])).to(device)
        pred = model_saved(data_input)
        # pred = torch.nn.functional.log_softmax(pred)
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = pred.cpu().detach().numpy()
        preds.extend(pred)
        labels.extend(label)
    metrics_in_file(labels, preds, bf_method, input_size, select_kind_62, f)


def do_test(
    f,
    X_test_path,
    y_test_path,
    input_size,
    folder_path,
    mode,
    bf_method,
    select_kind_62,
):
    ld_ts = LoadData(X_test_path, y_test_path, 0)
    X_test = ld_ts.get_X()
    y_test = ld_ts.get_y()
    test_dataset = GaitData(X_test, y_test)
    model_name = os.listdir(folder_path)[0]
    print("===================================================")
    print(f"Test of {model_name}.")
    path_model = os.path.join(folder_path, model_name)
    read_test_model(
        f,
        path_model,
        model_name,
        test_dataset,
        input_size,
        mode,
        bf_method,
        select_kind_62,
    )


if __name__ == "__main__":
    num_class = 3
    bfs_method = [
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
    models_name = ["gru", "lstm", "mlp"]
    modes = ["detection", "tracking"]
    inputs_best_feat_size = [60, 50, 40, 30, 20]  # [60, 50, 40, 30, 20, 10]
    inputs_manual_feat_size = [70, 34, 36, 58, 62]

    dir_ = f"results/2/200/metrics_res"
    os.makedirs(dir_, exist_ok=True)

    for mode in modes:
        base_dir_save_file = f"{dir_}/{mode}"
        os.makedirs(base_dir_save_file, exist_ok=True)
        for k, model_name in enumerate(models_name):
            indx = models_name.index(str(model_name))
            with open(
                f"{base_dir_save_file}/test_bfs_{mode}_{model_name}_metrics_table_{num_class}.txt",
                "w",
            ) as f:
                f.write("Methods, Features, Accuracy, Fscore, Precision, Recall\n")
                select_kind_62 = None
                bf_method = None
                # run to train on best features selected
                for input_best_feat_size in inputs_best_feat_size:
                    for bf_method in bfs_method:
                        train_dataset_path = f"./data/{mode}/best_feats/final_{input_best_feat_size}_best_feats_selections_from_70"
                        X_test_path = os.path.join(
                            train_dataset_path,
                            f"Xtest_{bf_method}_{input_best_feat_size}_selected_best_feats.File",
                        )
                        assert os.path.exists(
                            X_test_path
                        ), f"INFO: please check the path of {X_test_path} if exist."
                        y_test_path = os.path.join(train_dataset_path, f"ytest.File")
                        assert os.path.exists(
                            y_test_path
                        ), f"INFO: please check the path of {y_test_path} if exist."
                        folder_path = f"saved_models/{mode}/{model_name}_{bf_method}_{input_best_feat_size}"
                        do_test(
                            f,
                            X_test_path,
                            y_test_path,
                            input_best_feat_size,
                            folder_path,
                            mode,
                            bf_method,
                            select_kind_62,
                        )

            with open(
                f"{base_dir_save_file}/test_manual_{mode}_{model_name}_metrics_table_{num_class}.txt",
                "w",
            ) as f:
                f.write("Features, Accuracy, Fscore, Precision, Recall\n")
                # run to train on manual features generated
                for k, input_feat_size in enumerate(inputs_manual_feat_size):
                    bf_method = None
                    indx = models_name.index(str(model_name))
                    if input_feat_size == 62:
                        select_kinds_62 = ["rsa", "rsd", "rba"]
                        for select_kind_62 in select_kinds_62:
                            train_dataset_path = f"./data/{mode}/{input_feat_size}_{select_kind_62}_zero_padding_no"
                            X_test_path = os.path.join(
                                train_dataset_path, f"Xtest.File"
                            )
                            assert os.path.exists(
                                X_test_path
                            ), f"INFO: please check the path of {X_test_path} if exist."
                            y_test_path = os.path.join(
                                train_dataset_path, f"ytest.File"
                            )
                            folder_path = f"saved_models/{mode}/{model_name}_{input_feat_size}_{select_kind_62}"
                            do_test(
                                f,
                                X_test_path,
                                y_test_path,
                                input_feat_size,
                                folder_path,
                                mode,
                                bf_method,
                                select_kind_62,
                            )
                    else:
                        train_dataset_path = (
                            f"./data/{mode}/{input_feat_size}_zero_padding_no"
                        )
                        X_test_path = os.path.join(train_dataset_path, f"Xtest.File")
                        assert os.path.exists(
                            X_test_path
                        ), f"INFO: please check the path of {X_test_path} if exist."
                        y_test_path = os.path.join(train_dataset_path, f"ytest.File")
                        assert os.path.exists(
                            y_test_path
                        ), f"INFO: please check the path of {y_test_path} if exist."
                        folder_path = (
                            f"saved_models/{mode}/{model_name}_{input_feat_size}"
                        )
                        do_test(
                            f,
                            X_test_path,
                            y_test_path,
                            input_feat_size,
                            folder_path,
                            mode,
                            bf_method,
                            select_kind_62,
                        )
