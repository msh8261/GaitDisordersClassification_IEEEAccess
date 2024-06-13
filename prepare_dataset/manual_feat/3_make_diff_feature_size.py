# pylint: disable = c0103: invalid-name, c0116, e0602, w0621, w1514, e1121, e0401
import ast
import os
import shutil

import numpy as np
import torch

# import config.config_data as config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_size_data(x):
    """
    Get the all features input
    Retrun the tensor of size input_size
    """
    if input_size == 36:  # remove x,y features
        x = x[:, :, 34:]
    elif input_size == 34:  # only x,y features
        x = x[:, :, :34]
    elif input_size == 58:  # remove 12 dist points from nose
        x = x[:, :, :58]
    elif input_size == 62 and kind_62 == "rsa":  # remove 8 symetric angles
        x = torch.tensor(x)
        x1 = x[:, :, :34]
        key_points = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14])
        x2 = x[:, :, 34:50]
        x2 = torch.index_select(x2, 2, key_points)
        x3 = x[:, :, 50:]
        x = torch.dstack((x1, x2, x3)).numpy()
    elif input_size == 62 and kind_62 == "rba":  # remove 8 bones angles
        x = torch.tensor(x)
        x1 = x[:, :, :50]
        x2 = x[:, :, 58:]
        x = torch.dstack((x1, x2)).numpy()
    elif input_size == 62 and kind_62 == "rsd":  # remove 8 symetric dist
        x = torch.tensor(x)
        x1 = x[:, :, :34]
        key_points = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15])
        x2 = x[:, :, 34:50]
        x2 = torch.index_select(x2, 2, key_points)
        x3 = x[:, :, 50:]
        x = torch.dstack((x1, x2, x3)).numpy()
    else:
        raise ValueError("Wrong input size.")
    return x


def set_size_data(x):
    x = torch_size_data(x)
    x_str = [[[feats for feats in seq] for seq in batch] for batch in x]
    px = [item for sublist in x_str for item in sublist]
    px = [[str(item) for item in line] for line in px]
    px = [",".join(line) for line in px]
    return px


def write_to_file(
    X_train_path, X_test_path, dist_path, y_train_path, y_test_path
) -> None:
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
        path_xtrain_save = os.path.join(dist_path, "Xtrain.File")
        path_ytrain_save = os.path.join(dist_path, "ytrain.File")
        shutil.copy(y_train_path, path_ytrain_save)
        with open(path_xtrain_save, "w") as f:
            for line in x:
                f.write(line)
                f.write("\n")

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
        # print(np.array(Xtest).shape)
        x = set_size_data(Xtest)
        path_xtest_save = os.path.join(dist_path, "Xtest.File")
        path_ytest_save = os.path.join(dist_path, "ytest.File")
        shutil.copy(y_test_path, path_ytest_save)
        with open(path_xtest_save, "w") as f:
            for line in x:
                f.write(line)
                f.write("\n")


if __name__ == "__main__":
    import yaml

    with open("config/config_data.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    keypoints = 17
    # base_dir = "./Datasets/GaitAnalysis"
    dir_dist = config["dir_dist"]

    list_input_size = [34, 36, 58, 62]

    mode_folders = [name for name in os.listdir(dir_dist) if os.path.isdir(os.path.join(dir_dist, name))]

    list_mode_collection = mode_folders

    # only for 62 features
    list_kind_62 = ["rsa", "rsd", "rba"]

    for mode_collection in list_mode_collection:
        train_dataset_path = f"./data/{mode_collection}/70_zero_padding_no"
        os.makedirs(train_dataset_path, exist_ok=True)
        source = f"{dir_dist}/{mode_collection}"
        assert os.path.exists(
            os.path.join(source, "Xtrain.File")
        ), "INFO: please check if files exist in the source path!"
        dist = train_dataset_path
        shutil.copy(os.path.join(source, "Xtrain.File"), dist)
        shutil.copy(os.path.join(source, "ytrain.File"), dist)
        shutil.copy(os.path.join(source, "Xtest.File"), dist)
        shutil.copy(os.path.join(source, "ytest.File"), dist)

        X_train_path = os.path.join(train_dataset_path, "Xtrain.File")
        y_train_path = os.path.join(train_dataset_path, "ytrain.File")
        X_test_path = os.path.join(train_dataset_path, "Xtest.File")
        y_test_path = os.path.join(train_dataset_path, "ytest.File")

        for input_size in list_input_size:
            if input_size == 62:
                for kind_62 in list_kind_62:
                    dist_path = f"./data/{mode_collection}/{input_size}_{kind_62}_zero_padding_no"
                    os.makedirs(dist_path, exist_ok=True)
                    write_to_file(
                        X_train_path, X_test_path, dist_path, y_train_path, y_test_path
                    )
            else:
                dist_path = f"./data/{mode_collection}/{input_size}_zero_padding_no"
                os.makedirs(dist_path, exist_ok=True)
                write_to_file(
                    X_train_path, X_test_path, dist_path, y_train_path, y_test_path
                )
