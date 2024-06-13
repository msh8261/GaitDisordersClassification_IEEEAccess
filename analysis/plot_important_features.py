# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cycler

# import shutil


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
plt.rc("xtick", direction="out", color="gray", labelsize=12)
plt.rc("ytick", direction="out", color="gray", labelsize=12)
plt.rc("patch", edgecolor="#E6E6E6")
plt.rc("lines", linewidth=2, linestyle="-.")
matplotlib.rc("font", size=12)


def plot_important_features(
    methods_file,
    no_best_features,
    path_to_dataset,
    dir_base_to_best_features_files,
    save_files_path,
):
    fig = plt.figure(figsize=(8, 12))
    for k, method_file in enumerate(methods_file):
        plt.subplot(10, 5, k + 1)
        with open(path_to_dataset, "r") as fd:
            method_name = method_file.split("_")[1]
            feat_size = method_file.split("_")[2]
            print("==============================")
            print(method_name)
            print(feat_size)
            with open(
                os.path.join(dir_base_to_best_features_files, method_file), "r"
            ) as fb:
                best_feats = [l.strip() for l in fb.readlines()]
                best_feats = [[int(v) for v in l.split(",")] for l in best_feats]
                print(np.array(best_feats).shape)
                unique_feat, counts = np.unique(best_feats, return_counts=True)
                # print(unique_feat)
                # print(counts)

                colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
                fig = plt.figure(figsize=(8, 6))
                plt.bar(unique_feat, counts, color=colors[1], label=method_file)
                plt.legend()
                plt.xlabel("Feature size")
                plt.ylabel("Count")
                plt.tight_layout()
                # fig.savefig(fname=f'{save_files_path}/{method_name}_{feat_size}.png', dpi=600)
    fig.savefig(fname=f"{save_files_path}/all.pdf", format="pdf", bbox_inches="tight")
    # plt.show()
    plt.show(block=False)
    plt.pause(1)
    plt.close()


if __name__ == "__main__":
    # import yaml
    # with open('config/config_train.yaml', 'r') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    original_total_features = 70
    list_no_best_features = [20, 30, 40, 50, 60]

    list_mode_collection = ["detection", "tracking"]
    # list_mode_collection = ['detection']

    save_files_path = "results/2/200/important_features"
    os.makedirs(save_files_path, exist_ok=True)

    for mode_collection in list_mode_collection:
        fig = plt.figure(figsize=(18, 16))
        lbs = []
        for k, no_best_features in enumerate(list_no_best_features):
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

            methods_file = [
                f
                for f in os.listdir(dir_base_to_best_features_files)
                if f.endswith(".File")
            ]

            methods_file = [st for st in methods_file if st.split("_")[0] == "Xtrain"]
            methods = [method.split("_")[1] for method in methods_file]

            mode = "train"
            path_to_src_dataset = os.path.join(base_dir_data, f"X{mode}.File")
            path_to_src_y = os.path.join(base_dir_data, f"y{mode}.File")

            # plot_important_features(methods_file, no_best_features, path_to_src_dataset, dir_base_to_best_features_files, save_files_path)

            colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]

            with open(path_to_src_dataset, "r") as fd:
                for i, method_file in enumerate(methods_file):
                    plt.subplot(5, 10, (10 * k + i + 1))
                    method_name = method_file.split("_")[1]
                    feat_size = method_file.split("_")[2]
                    print("==============================")
                    print(method_name)
                    print(feat_size)
                    lbs.append(method_name)
                    labels = lbs[:10]
                    with open(
                        os.path.join(dir_base_to_best_features_files, method_file), "r"
                    ) as fb:
                        best_feats = [l.strip() for l in fb.readlines()]
                        best_feats = [
                            [int(v) for v in l.split(",")] for l in best_feats
                        ]
                        print(np.array(best_feats).shape)
                        unique_feat, counts = np.unique(best_feats, return_counts=True)
                        # print(unique_feat)
                        # print(counts)
                        plt.bar(unique_feat, counts, color=colors[i], label=method_name)
                        if k == 4:
                            plt.legend(
                                loc="upper left",
                                bbox_to_anchor=(-0.25, -0.25),
                                fancybox=True,
                                shadow=True,
                            )

                        if i == 0:
                            plt.ylabel("Count", fontsize=12)
                            plt.xlabel("Features", fontsize=12)

                        if k == 0 and i == 0:
                            if mode_collection == "detection":
                                mode = "Detection"
                            else:
                                mode = "Tracking"
                            plt.title(f"{mode}  Results", fontsize=20)
                        if i == 4:
                            plt.title(f"{feat_size} selected features", fontsize=16)

        print(labels)
        # plt.legend(loc='best', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=10)
        plt.tight_layout()
        fig.savefig(
            fname=f"{save_files_path}/{mode_collection}_all.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.show(block=False)
        plt.pause(1)
        plt.close()
