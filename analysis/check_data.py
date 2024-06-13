# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103
import os

import numpy as np

import config.config_train as config

sequences = config.params["sequences"]


def test_x(X_train_path, info=""):
    file = open(X_train_path[0], "r")
    X = np.array([elem.split(",") for elem in file], dtype=np.float32)
    file.close()
    blocks = int(len(X) / (sequences))
    X_ = np.array(np.split(X, blocks))
    print("=========================================")
    print(info)
    print("=========================================")
    # print(X_)
    print(X_.shape)


def test_y(y_test_path, info=""):
    file = open(y_test_path[0], "r")
    y = np.array(
        [elem for elem in [row.replace("  ", " ").strip().split(" ") for row in file]],
        dtype=np.int32,
    )
    file.close()
    print("=========================================")
    print(info)
    print("=========================================")
    # print(y - 1)
    print(y.shape)


if __name__ == "__main__":
    # train_dataset_path = config.params["train_dataset_path"]
    # sequences = config.params["sequences"]

    # train_dataset_path = r'C:\Users\mohsen\Desktop\Postdoc_Upa\Datasets\GaitAnalysis\dst'

    train_dataset_path = "data/tracking/34_zero_padding_no"

    X_train_path = os.path.join(train_dataset_path, "Xtrain.File")
    y_train_path = os.path.join(train_dataset_path, "ytrain.File")
    X_test_path = os.path.join(train_dataset_path, "Xtest.File")
    y_test_path = os.path.join(train_dataset_path, "ytest.File")

    inputs_path_zipped = [(X_train_path, X_test_path, y_train_path, y_test_path)]
    # inputs_path_zipped = [(X_train_path, y_train_path)]

    X_train_path, X_test_path, y_train_path, y_test_path = list(
        zip(*inputs_path_zipped)
    )
    test_x(X_train_path, "INFO: TEST Xtrain")
    test_y(y_train_path, "INFO: TEST ytrain")
    test_x(X_test_path, "INFO: TEST Xtest")
    test_y(y_test_path, "INFO: TEST ytest")
