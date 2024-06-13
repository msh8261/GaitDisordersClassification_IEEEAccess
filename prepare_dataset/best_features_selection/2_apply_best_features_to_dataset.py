# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103
import os
import shutil
import yaml
# import numpy as np
with open("config/config_train.yaml", "r") as f:
    config_tr = yaml.load(f, Loader=yaml.FullLoader)



def write_to_file(
    method_file,
    mode,
    no_best_features,
    path_to_dataset,
    dir_base_to_best_features_files,
    dir_to_save_best_features_file,
):
    with open(path_to_dataset, "r") as fd:
        method_name = method_file.split("_")[1]
        print("==============================")
        print(method_name)
        with open(
            os.path.join(dir_base_to_best_features_files, method_file), "r"
        ) as fb:
            best_feats = fb.readlines()
            dataset = fd.readlines()
            len_best_feats = len(best_feats)
            len_dataset = len(dataset)
            print("lines of best feature file: ", len_best_feats)
            print("lines of dataset: ", len_dataset)
            # if len(best_feats) != len(dataset):
            #     raise "INFO: the lines of best features file and dataset must be equal."
            path_save_file = os.path.join(
                dir_to_save_best_features_file,
                f"X{mode}_{method_name}_{no_best_features}_selected_best_feats.File",
            )
            with open(path_save_file, "w") as fw:
                j = 0
                for i in range(len(dataset)):
                    # print("line: ", i)
                    l1 = dataset[i].split(",")
                    if j >= len_best_feats:
                        j = 0
                    l2 = best_feats[j].split(",")
                    selected_best_feats = [float(l1[int(ind)]) for ind in l2]
                    arr = [str(item) for item in selected_best_feats]
                    arr = ["%s" % item for item in arr]
                    final_arr_to_file = ",".join(arr)
                    fw.write(final_arr_to_file)
                    fw.write("\n")
                    j += 1


if __name__ == "__main__":    

    with open("config/config_data.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dir_dist = config["dir_dist"]

    mode_folders = [name for name in os.listdir(dir_dist) if os.path.isdir(os.path.join(dir_dist, name))]

    list_mode_collection = mode_folders

    original_total_features = 70
    list_no_best_features = list_no_best_features = config_tr["input_data_params"]["inputs_best_feat_size"]

    for mode_collection in list_mode_collection:
        for no_best_features in list_no_best_features:
            base_dir_data = (
                f"./data/{mode_collection}/{original_total_features}_zero_padding_no"
            )
            assert os.path.exists(
                base_dir_data
            ), f"INFO: please check the path of {base_dir_data} if exist."

            dir_base_to_best_features_files = f"data/{mode_collection}/best_feats/{no_best_features}_best_feats_selections_from_{original_total_features}"
            assert os.path.exists(
                dir_base_to_best_features_files
            ), f"INFO: please check the path of {dir_base_to_best_features_files} if exist."

            dir_to_save_best_features_file = f"data/{mode_collection}/best_feats/final_{no_best_features}_best_feats_selections_from_{original_total_features}"
            os.makedirs(dir_to_save_best_features_file, exist_ok=True)

            methods_file = [
                f
                for f in os.listdir(dir_base_to_best_features_files)
                if f.endswith(".File")
            ]

            methods_file = [st for st in methods_file if st.split("_")[0] == "Xtrain"]

            mode = "train"
            path_to_src_dataset = os.path.join(base_dir_data, f"X{mode}.File")
            path_to_src_y = os.path.join(base_dir_data, f"y{mode}.File")
            path_y_save = os.path.join(dir_to_save_best_features_file, f"y{mode}.File")
            shutil.copy(path_to_src_y, path_y_save)
            for method_file in methods_file:
                write_to_file(
                    method_file,
                    mode,
                    no_best_features,
                    path_to_src_dataset,
                    dir_base_to_best_features_files,
                    dir_to_save_best_features_file,
                )

            mode = "test"
            path_to_src_dataset = os.path.join(base_dir_data, f"X{mode}.File")
            path_to_src_y = os.path.join(base_dir_data, f"y{mode}.File")
            path_y_save = os.path.join(dir_to_save_best_features_file, f"y{mode}.File")
            shutil.copy(path_to_src_y, path_y_save)
            for method_file in methods_file:
                write_to_file(
                    method_file,
                    mode,
                    no_best_features,
                    path_to_src_dataset,
                    dir_base_to_best_features_files,
                    dir_to_save_best_features_file,
                )
