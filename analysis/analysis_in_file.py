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

# import config.config_train as config
# from src.dataset import GaitData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

keypoints = 17

show_each_fig = False
analysis_own = False

if analysis_own:
    input_size = 70
else:
    input_size = 60


def set_size_data(x):
    x_str = [[[(c) for c in a] for a in b] for b in x]
    px = [item for sublist in x_str for item in sublist]
    px = [[str(item) for item in line] for line in px]
    px = [",".join(line) for line in px]
    return px


def analysis_on_features(lines):
    individuals = [[], [], [], []]
    xs = []
    ys = []
    ds = []
    angs = []
    for line in lines:
        features = line
        features = features.split(",")
        features = [(float(j)) for j in features]
        if analysis_own:
            features1 = features[: (keypoints) * 2]
            blocks1 = int(len(features1) / 2)
            features2 = features[(keypoints) * 2 : (keypoints) * 2 + 16]
            blocks2 = int(len(features2) / 2)
            features3 = features[(keypoints) * 2 + 16 : (keypoints) * 2 + 24]
            blocks3 = int(len(features3) / 1)
            features4 = features[(keypoints) * 2 + 24 :]
            blocks4 = int(len(features4) / 1)
            features1 = np.array(np.split(np.array(features1), blocks1))
            features2 = np.array(np.split(np.array(features2), blocks2))
            features3 = np.array(np.split(np.array(features3), blocks3))
            features4 = np.array(np.split(np.array(features4), blocks4))
            x = np.array([fea[0] for fea in features1])
            y = np.array([fea[1] for fea in features1])
            d = np.array([fea for fea in features4])
            ang2 = np.array([fea for fea in features3])
        else:
            features1 = features[: (keypoints) * 2]
            blocks1 = int(len(features1) / 2)
            features2 = features[(keypoints) * 2 : (keypoints) * 2 + 18]
            blocks2 = int(len(features2) / 1)
            features3 = features[(keypoints) * 2 + 18 :]
            blocks3 = int(len(features3) / 1)
            features1 = np.array(np.split(np.array(features1), blocks1))
            features2 = np.array(np.split(np.array(features2), blocks2))
            features3 = np.array(np.split(np.array(features3), blocks3))
            x = np.array([fea[0] for fea in features1])
            y = np.array([fea[1] for fea in features1])
            d = np.array([fea for fea in features2])
            ang2 = np.array([fea for fea in features3])

        # x = round(x.mean(), 2)
        # y = round(y.mean(), 2)
        # d = round(d.mean(), 2)
        x = x.mean()
        y = y.mean()
        d = d.mean()
        ang2 = ang2.mean()

        # print(x, y, d, ang2)
        xs.append(x)
        ys.append(y)
        ds.append(d)
        angs.append(ang2)

    # if show_each_fig:
    #     fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    #     axs[0][0].plot(xs, label="x")
    #     axs[0][0].legend()
    #     axs[0][1].plot(ys, label="y")
    #     axs[0][1].legend()
    #     axs[1][0].plot(angs, label="angle")
    #     axs[1][0].legend()
    #     axs[1][1].plot(ds, label="distance")
    #     axs[1][1].legend()
    #     plt.show()

    if show_each_fig:
        N = len(xs)
        fig, axs = plt.subplots(3, 2, figsize=(8, 6))
        axs[0][0].scatter(xs, ys, label="x,y", c=np.random.rand(N), alpha=0.5)
        axs[0][0].legend()
        axs[0][1].scatter(xs, ds, label="y,d", c=np.random.rand(N), alpha=0.5)
        axs[0][1].legend()
        axs[1][0].scatter(xs, angs, label="x,angle", c=np.random.rand(N), alpha=0.5)
        axs[1][0].legend()
        axs[1][1].scatter(ys, ds, label="y,d", c=np.random.rand(N), alpha=0.5)
        axs[1][1].legend()
        axs[2][0].scatter(ys, angs, label="y,d", c=np.random.rand(N), alpha=0.5)
        axs[2][0].legend()
        axs[2][1].scatter(ds, angs, label="d,angle", c=np.random.rand(N), alpha=0.5)
        axs[2][1].legend()
        plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    n, bins, patches = axs[0][0].hist(
        (xs), color="r", label="mean of x values in all sessions"
    )
    axs[0][0].legend(loc="upper left")
    n, bins, patches = axs[0][1].hist(
        (ys), color="b", label="mean of y values in all sessions"
    )
    axs[0][1].legend(loc="upper left")
    n, bins, patches = axs[1][0].hist(
        (ds), color="c", label="mean of distances in all sessions"
    )
    axs[1][0].legend(loc="upper left")
    n, bins, patches = axs[1][1].hist(
        (angs), color="k", label="mean of angles in all sessions"
    )
    axs[1][1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("results/figure_cls_all", bbox_inches="tight", dpi=600)
    plt.show()


if __name__ == "__main__":
    if analysis_own:
        train_dataset_path = "./data/final/100_zero_padding_no"
    else:
        train_dataset_path = "./data/final/100_zero_padding_no_honza"

    X_train_path = os.path.join(train_dataset_path, "Xtrain.File")

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
        print(np.array(Xtrain).shape)
        x = set_size_data(Xtrain)
        with open("Xdata.File", "w") as f:
            for line in x:
                f.write(line)
                f.write("\n")

    with open("Xdata.File", "r") as f:
        file = f.readlines()
        analysis_on_features(file)
