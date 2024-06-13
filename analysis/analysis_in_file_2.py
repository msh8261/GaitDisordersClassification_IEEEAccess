import ast
import os

import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# import pytorch_lightning as pl
# import seaborn as sns
import torch
# import torch.nn as nn
# import torch.nn.functional as F
from sklearn.manifold import TSNE

import config.config_train as config
from prepare_dataset.decorators import logger

# import random


# from prepare_dataset.decorators import countcall, dataclass, timeit
# from src.dataset import GaitData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

keypoints = 17


def set_size_data(x):
    x_str = [[[(c) for c in a] for a in b] for b in x]
    px = [item for sublist in x_str for item in sublist]
    px = [[str(item) for item in line] for line in px]
    px = [",".join(line) for line in px]
    return px


@logger
def show_clusters_of_data(data_lines, label_lines, random_state):
    X = np.array([elem.split(",") for elem in data_lines], dtype=np.float32)
    blocks = int(len(X) / config.params["sequences"])
    data_vec = np.array(np.split(X, blocks))
    data_vec = np.array([[np.std(arr) for arr in block] for block in data_vec])
    y = np.array(
        [
            elem
            for elem in [
                row.replace("  ", " ").strip().split(" ") for row in label_lines
            ]
        ],
        dtype=np.int32,
    )
    classes = [val - 1 for val in y]
    tsne = TSNE(n_components=2, learning_rate="auto", random_state=random_state)
    clustered = tsne.fit_transform(data_vec)
    fig = plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("Spectral", config.params["num_class"])
    plt.scatter(*zip(*clustered), c=classes, cmap=cmap)
    plt.colorbar(drawedges=True)
    plt.show()


@logger
def analysis_statistic_on_features(lines):
    list_feats = []
    list_feats_mean = []
    list_feats_std = []
    for line in lines:
        features = line
        features = features.split(",")
        features = np.array([(float(val)) for val in features])
        features_mean = features.mean()
        features_std = features.std()
        list_feats.append(features)
        list_feats_mean.append(features_mean)
        list_feats_std.append(features_std)

    corr = np.corrcoef(np.array(list_feats).T)
    # print((corr.shape))

    # fig = plt.figure(figsize=(30, 20))
    # sns.heatmap(corr, linewidths=.2, annot=True, cmap='RdBu')
    # fig.tight_layout()
    # plt.show()

    # plt.subplots_adjust(left=0.1, bottom=0.15, right=0.99, top=0.95)
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(
        corr, cmap=plt.cm.get_cmap("Reds"), interpolation="nearest", aspect="auto"
    )
    plt.colorbar()
    fig.tight_layout()
    plt.show()

    # cols = corr.shape[1]
    # if corr.shape[1]>30:
    #     cols = int(corr.shape[1]/2)
    # for i in range(int(corr.shape[0]/10)):
    #     fig = plt.figure(figsize=(16, 6))
    #     sns.heatmap(corr[10*i:10*(i+1), :cols], annot=True, cmap='RdBu')
    #     fig.tight_layout()
    #     fig = plt.figure(figsize=(16, 6))
    #     sns.heatmap(corr[10*i:10*(i+1), cols:cols*2], annot=True, cmap='RdBu')
    #     fig.tight_layout()
    #     plt.show()

    # g = sns.clustermap(corr,
    #                method = 'complete',
    #                cmap   = 'RdBu',
    #                annot  = True,)
    #                #annot_kws = {'size': 8})
    # plt.setp(g.ax_heatmap.get_xticklabels(), rotation=60)
    # plt.show()

    fig = plt.figure(figsize=(6, 4))
    plt.hist(
        (list_feats_mean), color="b", label="mean of features values in all sessions"
    )
    plt.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("results/figure_mean_feats", bbox_inches="tight", dpi=600)
    plt.show()

    fig = plt.figure(figsize=(6, 4))
    plt.hist(
        (list_feats_std), color="g", label="std of features values in all sessions"
    )
    plt.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("results/figure_std_feats", bbox_inches="tight", dpi=600)
    plt.show()


def write_to_file(X_train_path, save_file_name="Xdata.File"):
    with open(X_train_path, "r") as f:
        Xtrain = []
        for line in f:
            if len(line.split()) > 0:
                str_line = line.split()[0]
                list_line = list(ast.literal_eval(str_line))
                Xtrain.append(list_line)
            else:
                print("INFO: Error of empty data!")
        Xtrain = np.array(Xtrain)
        Xtrain = Xtrain.reshape(1, Xtrain.shape[0], Xtrain.shape[1])
        # print(np.array(Xtrain).shape)
        x = set_size_data(Xtrain)
        with open(save_file_name, "w") as f:
            for line in x:
                f.write(line)
                f.write("\n")


@logger
def analysis_features(X_train_path):
    print(f"file: {X_train_path}")
    with open(X_train_path, "r") as f:
        data = f.readlines()
        analysis_statistic_on_features(data)


@logger
def analysis_clusters(X_train_path, y_train_path, random_state):
    print(f"file: {X_train_path}")
    with open(X_train_path, "r") as f:
        data = f.readlines()
        with open(y_train_path, "r") as f:
            labels = f.readlines()
            show_clusters_of_data(data, labels, random_state)


if __name__ == "__main__":
    # 104 or 70
    original_feats = 70
    # True for best feature selections
    best_feats = True
    # 10 to 70
    input_size = 50

    if best_feats:
        train_dataset_path = f"./data/best_feats/final_{input_size}_best_feats_selections_from_{original_feats}"
        bf_methods = [
            f"ANOVA_{input_size}_selected_best_feats",
            f"chi2_{input_size}_selected_best_feats",
            f"cmim_{input_size}_selected_best_feats",
            f"disr_{input_size}_selected_best_feats",
            f"kruskal_{input_size}_selected_best_feats",
            f"mifs_{input_size}_selected_best_feats",
            f"MultiSURF_{input_size}_selected_best_feats",
            f"ReliefF_{input_size}_selected_best_feats",
            f"SURF_{input_size}_selected_best_feats",
            f"SURFstar_{input_size}_selected_best_feats",
        ]
        bf_method = bf_methods[0]
        X_train_path = os.path.join(train_dataset_path, f"Xtrain_{bf_method}.File")
        y_train_path = os.path.join(train_dataset_path, f"ytrain.File")
    else:
        train_dataset_path = f"./data/{original_feats}_zero_padding_no"
        X_train_path = os.path.join(train_dataset_path, f"Xtrain.File")
        y_train_path = os.path.join(train_dataset_path, f"ytrain.File")

    # write_to_file(X_train_path, save_file_name='Xdata.File')

    analysis_features(X_train_path)

    random_state = 2
    analysis_clusters(X_train_path, y_train_path, random_state)
