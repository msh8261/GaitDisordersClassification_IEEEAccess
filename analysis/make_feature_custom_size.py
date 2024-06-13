import ast
import os

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

keypoints = 17

analysis_own = True

if analysis_own:
    input_size = 70
else:
    input_size = 60


def torch_size_data(x):
    if input_size == 70:
        key_points = [
            0,
            1,
            4,
            5,
            8,
            9,
            12,
            13,
            16,
            17,
            20,
            21,
            24,
            25,
            28,
            29,
            32,
            33,
            36,
            37,
            40,
            41,
            44,
            45,
            48,
            49,
            52,
            53,
            56,
            57,
            60,
            61,
            64,
            65,
        ]
        x1 = torch.tensor(x[:, :, :68])
        x1 = torch.index_select(x1, 2, torch.tensor(key_points))
        x2 = torch.tensor(x[:, :, 68:])
        x = torch.dstack((x1, x2))
    elif input_size == 62:
        key_points = [
            0,
            1,
            4,
            5,
            8,
            9,
            12,
            13,
            16,
            17,
            20,
            21,
            24,
            25,
            28,
            29,
            32,
            33,
            36,
            37,
            40,
            41,
            44,
            45,
            48,
            49,
            52,
            53,
            56,
            57,
            60,
            61,
            64,
            65,
        ]
        x1 = torch.tensor(x[:, :, :68])
        x1 = torch.index_select(x1, 2, torch.tensor(key_points))
        key_points = [0, 2, 4, 6, 8, 10, 12, 14]
        x2 = torch.tensor(x[:, :, 68:84])
        x2 = torch.index_select(x2, 2, torch.tensor(key_points))
        x3 = torch.tensor(x[:, :, 84:])
        x = torch.dstack((x1, x2, x3))
    return x


def numpy_size_data(x):
    x = x.round(4)
    key_points = [
        0,
        1,
        4,
        5,
        8,
        9,
        12,
        13,
        16,
        17,
        20,
        21,
        24,
        25,
        28,
        29,
        32,
        33,
        36,
        37,
        40,
        41,
        44,
        45,
        48,
        49,
        52,
        53,
        56,
        57,
        60,
        61,
        64,
        65,
    ]
    x1 = x[:, :, :68]
    x1 = np.array(x1)
    x1 = [np.take(arr, np.array(key_points)) for arr in x1[0]]
    x1 = np.array(x1)
    x1 = x1.reshape(1, x1.shape[0], x1.shape[1])
    x2 = x[:, :, 68:]
    x2 = np.array(x2)
    x = np.dstack((x1, x2))
    return x


def set_size_data(x):
    if analysis_own:
        xt = torch_size_data(x)
        x = numpy_size_data(x)
        # print(x)
        print(xt)
        x_str = [[[(c) for c in a] for a in b] for b in x]
        px = [item for sublist in x_str for item in sublist]
        px = [[str(item) for item in line] for line in px]
        px = [",".join(line) for line in px]
        # print(x[0, 0, :])
        # print(px[0])
    else:
        x_str = [[[(c) for c in a] for a in b] for b in x]
        px = [item for sublist in x_str for item in sublist]
        px = [[str(item) for item in line] for line in px]
        px = [",".join(line) for line in px]

    return px


if __name__ == "__main__":
    if analysis_own:
        train_dataset_path = "./data/final/100_zero_padding_no"
    else:
        train_dataset_path = "./data/final/100_zero_padding_no_honza"

    X_train_path = os.path.join(train_dataset_path, "Xtrain.File")
    X_test_path = os.path.join(train_dataset_path, "Xtest.File")

    dist_path = "./data/final/100_zero_padding_no_self"

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
        # path_train_save = os.path.join(dist_path, 'Xtrain.File')
        # with open(path_train_save, 'w') as f:
        #     for line in x:
        #         f.write(line)
        #         f.write("\n")

    with open(X_test_path, "r") as f:
        Xtest = []
        for line in f:
            if len(line.split()) > 0:
                str_line = line.split()[0]
                list_line = list(ast.literal_eval(str_line))
                Xtest.append(list_line)
            else:
                print("INFO: Error of empty data!")
        Xtest = np.array(Xtest)
        Xtest = Xtest.reshape(1, Xtest.shape[0], Xtest.shape[1])
        print(np.array(Xtest).shape)
        x = set_size_data(Xtest)
        # path_test_save = os.path.join(dist_path, 'Xtest.File')
        # with open(path_test_save, 'w') as f:
        #     for line in x:
        #         f.write(line)
        #         f.write("\n")
