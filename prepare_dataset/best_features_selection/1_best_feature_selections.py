# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103
import os

import numpy as np
# import pandas as pd
import yaml

from prepare_dataset.best_features_selection.FSM import FSM
from prepare_dataset.tools.decorators import logger # , timeit
# from src.dataset import GaitData
from src.load import LoadData

with open("config/config_train.yaml", "r") as f:
    config_tr = yaml.load(f, Loader=yaml.FullLoader)

@logger
def show_resutls(result, method_name):
    print(f"=================== results of {method_name} ===================")
    print(f"best featurs: {result}")
    if result is not None:
        print(f"number of best features: {len(result)}")


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


@logger
def get_data(dataset_path, mode):
    X_path = os.path.join(dataset_path, f"X{mode}.File")
    y_path = os.path.join(dataset_path, f"y{mode}.File")
    ld = LoadData(X_path, y_path, config_tr["input_data_params"]["num_augmentation"], False)
    X = ld.get_X()
    y = ld.get_y()
    return X, y


@logger
def write_to_file(
    base_dir, list_method_name, lists_all, sequences, no_best_features, mode
):
    for method_name in list_method_name:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(method_name)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
        file_save_path = os.path.join(
            base_dir, f"X{mode}_{method_name}_{no_best_features}_best_feats.File"
        )
        with open(file_save_path, "w") as f:
            condition1 = all(
                len(lists_all[method_name][i].split(",")) == no_best_features
                for i in range(sequences)
            )
            condition2 = len(lists_all[method_name]) == sequences
            print(condition1)
            print(condition2)
            print(len(lists_all[method_name]), sequences)
            if condition1 and condition2:
                for feats in lists_all[method_name]:
                    f.write(feats)
                    f.write("\n")
                print(f"features of {method_name} written in the file.")
            else:
                raise Exception("INFO: conditions are not proved please check them.")


def get_list_of_features_index(
    dataset_path, mode, selectors, list_method_name, lists_all, no_best_features
):
    X_, y_ = get_data(dataset_path, mode)
    sequences_ = X_.shape[1] 
    for i in range(sequences_):
        X = np.array([x[i] for x in X_])
        y = np.array([int(val) for val in y_])
        # B is the number of bootstrap subset to sample.
        results = selectors.Bootstrapper(X, y, B=2, Sample_size=X_.shape[0])
        for method_name in list_method_name:
            method_res = list(results[method_name][0])
            # show_resutls(method_res, method_name)
            feats = clean_arr_save_to_file(method_res, no_best_features)
            lists_all[method_name].append(feats)
    return lists_all, sequences_


if __name__ == "__main__":
    with open("config/config_data.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # keypoints = 17

    dir_dist = config["dir_dist"]

    mode_folders = [name for name in os.listdir(dir_dist) if os.path.isdir(os.path.join(dir_dist, name))]

    list_mode_collection = mode_folders

    original_total_features = 70
    list_no_best_features = config_tr["input_data_params"]["inputs_best_feat_size"]


    for mode_collection in list_mode_collection:
        for no_best_features in list_no_best_features:
            dataset_path = (
                f"./data/{mode_collection}/{original_total_features}_zero_padding_no"
            )
            assert os.path.exists(
                dataset_path
            ), f"INFO: please check the path of {dataset_path} if exist."

            base_dist_dir = f"./data/{mode_collection}/best_feats/{no_best_features}_best_feats_selections_from_{original_total_features}"
            os.makedirs(base_dist_dir, exist_ok=True)

            # k is number of best features
            selectors = FSM(k=no_best_features, filler=-1)

            list_method_name = config_tr["model_params"]["bfs_methods"]
   
            lists_list_train = [[] for i in range(len(list_method_name))]
            lists_all_train = dict(zip(list_method_name, lists_list_train))

            mode = "train"
            lists_feats_ind, sequences = get_list_of_features_index(
                dataset_path,
                mode,
                selectors,
                list_method_name,
                lists_all_train,
                no_best_features,
            )
            write_to_file(
                base_dist_dir,
                list_method_name,
                lists_feats_ind,
                sequences,
                no_best_features,
                mode,
            )
