# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103, e1101
import os
from itertools import cycle

import cv2
import matplotlib
import matplotlib.pyplot as plt
# import glob
import numpy as np
# import seaborn as sn
import pandas as pd
import torch
# import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn import metrics
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import classification_report
from sklearn.metrics import (PrecisionRecallDisplay, accuracy_score,
                             average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve)
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import label_binarize
# from random import shuffle
from sklearn.utils import shuffle

# from pretty_confusion_matrix import pp_matrix
from analysis.pp_matrix import pp_matrix
# from prepare_dataset.tools.decorators import (countcall, dataclass, timeit)
from prepare_dataset.tools.decorators import logger
from src.dataset import GaitData
# import config.config_train as config
from src.load import LoadData
from matplotlib import cycler

colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', titlesize=14,
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray', labelsize=14)
plt.rc('ytick', direction='out', color='gray', labelsize=14)
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2, linestyle='-.')
plt.rcParams["figure.figsize"] = (8,6)
matplotlib.rc('font', size=14)
matplotlib.rc('lines', linewidth=3, linestyle='-.')
colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])


device = torch.device("cuda" if torch.cuda.is_available() else "cup")


def filters(im, mode="sharpen"):
    # remove noise
    im = cv2.GaussianBlur(im, (3, 3), 0)
    if mode == "laplacian":
        # convolute with proper kernels
        im_out = cv2.Laplacian(im, cv2.CV_32F)
    elif mode == "sobelx":
        im_out = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)  # x
    elif mode == "sobely":
        im_out = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=5)  # y
    elif mode == "sharpen":
        # kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        im_out = cv2.filter2D(src=im, ddepth=-1, kernel=kernel)
    return im_out


def convert_to_spect(img):
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    phase_spectrum = np.angle(dft_shift)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift))
    return magnitude_spectrum


def plot_precision_recall(
    recall, precision, f_scores, average_precision, save_file_name
):

    _, ax = plt.subplots(figsize=(7, 8))

    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, color="gold")
    _ = display.ax_.set_title("Micro-averaged over all classes")

    for i, color in zip(range(num_class), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {i+1}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class")

    fig = display.figure_
    fig.tight_layout()
    fig.savefig(save_file_name, dpi=600)

    plt.show(block=False)
    plt.pause(1)
    plt.close()


def save_accuracy_in_file(labels_gt, labels, model_name, input_size, mode):
    base_dir_save_file = f"results/test_res/{mode}"
    dir_save_metrics_fig = f"{base_dir_save_file}/metrics"
    os.makedirs(base_dir_save_file, exist_ok=True)
    os.makedirs(dir_save_metrics_fig, exist_ok=True)
    with open(
        f"{base_dir_save_file}/test_results_cls"
        + str(num_class)
        + f"_{model_name}_{input_size}_{mode}.txt",
        "w",
    ) as f:
        acc = accuracy_score(labels_gt, labels)
        print(f"Test accuracy is: {acc.round(2)}")
        f.write(f"Test accuracy is: {acc.round(2)} \n")

        precision, recall, fscore, support = score(labels_gt, labels)

        print("precision: {} ".format(precision.round(2)))
        f.write("precision: {} \n".format(precision.round(2)))
        print("recall: {}".format(recall.round(2)))
        f.write("recall: {} \n".format(recall.round(2)))
        print("fscore: {}".format(fscore.round(2)))
        f.write("fscore: {} \n".format(fscore.round(2)))
        print("support: {}".format(support))
        f.write("support: {} \n".format(support))

        print("ave f1_score: ", f1_score(labels_gt, labels, average="macro").round(2))
        f.write(
            "ave f1_score: {} \n".format(
                f1_score(labels_gt, labels, average="macro").round(2)
            )
        )
        print(
            "ave precision: ",
            precision_score(labels_gt, labels, average="macro").round(2),
        )
        f.write(
            "ave precision: {} \n".format(
                precision_score(labels_gt, labels, average="macro").round(2)
            )
        )
        print(
            "ave recall: {}".format(
                recall_score(labels_gt, labels, average="macro").round(2)
            )
        )
        f.write(
            "ave recall: {} \n".format(
                recall_score(labels_gt, labels, average="macro").round(2)
            )
        )

        conf_mat = confusion_matrix(labels_gt, labels)
        print(conf_mat)

        df_cm = pd.DataFrame(conf_mat, index=range(1, 4), columns=range(1, 4))
        cmap = "tab20b"  # 'gist_yarg' #'gnuplot' #'gist_yarg' #'gnuplot'  #cmap=plt.cm.Blues #'coolwarm_r' #'PuRd'

        pp_matrix(
            df_cm,
            cmap=cmap,
            fz=14,
            lw=0.5,
            figsize=[8, 6],
            save_file_name=f"{dir_save_metrics_fig}/{model_name}_{input_size}_{mode}_metrics_confusion.png",
        )

        # average_precision = [round(float(val), 2) for val in precision]
        # precision = [round(float(val), 2)  for val in precision]
        # recall = [round(float(val), 2)  for val in recall]
        precision = dict()
        recall = dict()
        average_precision = dict()

        labels_gt = label_binarize(labels_gt, classes=[*range(num_class)])
        labels = label_binarize(labels, classes=[*range(num_class)])

        for i in range(num_class):
            precision[i], recall[i], _ = precision_recall_curve(
                labels_gt[:, i], labels[:, i]
            )
            average_precision[i] = average_precision_score(
                labels_gt[:, i], labels[:, i]
            )

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            labels_gt.ravel(), labels.ravel()
        )
        average_precision["micro"] = average_precision_score(
            labels_gt, labels, average="micro"
        )

        save_file_name = f"{dir_save_metrics_fig}/{model_name}_{input_size}_{mode}_precision_recall.png"
        plot_precision_recall(
            recall, precision, fscore, average_precision, save_file_name
        )


def test_saved_model(path_model, model_name, test_dataset, input_size, mode):
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

    print(labels)
    print(preds)
    acc = metrics.accuracy_score(labels, preds)
    print(f"---->>>> acc: {acc} <<<<-------")

    save_accuracy_in_file(labels, preds, model_name, input_size, mode)


@logger
def do_test(X_test_path, y_test_path, input_size, folder_path, mode):
    ld_ts = LoadData(X_test_path, y_test_path, 0)
    X_test = ld_ts.get_X()
    y_test = ld_ts.get_y()
    test_dataset = GaitData(X_test, y_test)
    model_name = os.listdir(folder_path)[0]
    print("===================================================")
    print(f"Test of {model_name}.")
    path_model = os.path.join(folder_path, model_name)
    test_saved_model(path_model, model_name, test_dataset, input_size, mode)


if __name__ == "__main__":
    with open("config/config_train.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    num_class = config["input_data_params"]["num_class"]

    bfs_method = config["model_params"]["bfs_methods"]
    models_name = config["model_params"]["models_name"]

    modes = config["model_params"]["modes"]
    inputs_best_feat_size = config["input_data_params"]["inputs_best_feat_size"]
    inputs_manual_feat_size = config["input_data_params"]["inputs_manual_feat_size"]

    for mode in modes:
        for model_name in models_name:
            select_kind_62 = None
            bf_method = None
            # run to train on best features selected
            # inputs_best_feat_size = [60, 50, 40, 30, 20, 10]
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
                        X_test_path,
                        y_test_path,
                        input_best_feat_size,
                        folder_path,
                        mode,
                    )

            # run to train on manual features generated
            # inputs_manual_feat_size = [70, 34, 36, 58, 62]
            for input_feat_size in inputs_manual_feat_size:
                if input_feat_size == 62:
                    select_kinds_62 = ["rsa", "rsd", "rba"]
                    for select_kind_62 in select_kinds_62:
                        train_dataset_path = f"./data/{mode}/{input_feat_size}_{select_kind_62}_zero_padding_no"
                        X_test_path = os.path.join(train_dataset_path, f"Xtest.File")
                        assert os.path.exists(
                            X_test_path
                        ), f"INFO: please check the path of {X_test_path} if exist."
                        y_test_path = os.path.join(train_dataset_path, f"ytest.File")
                        folder_path = f"saved_models/{mode}/{model_name}_{input_feat_size}_{select_kind_62}"
                        do_test(
                            X_test_path, y_test_path, input_feat_size, folder_path, mode
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
                    folder_path = f"saved_models/{mode}/{model_name}_{input_feat_size}"
                    do_test(
                        X_test_path, y_test_path, input_feat_size, folder_path, mode
                    )
