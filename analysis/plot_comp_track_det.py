# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103, e1101
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
# import seaborn as sn
from matplotlib import cycler

# import sys
# from itertools import cycle


# import ast
# import glob


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


def arrange_metrics_kfold(path):
    list_values = []
    list_keys = []
    dic_data = {}
    num_metrics = 22
    with open(path, "r") as f:
        file = f.read().split("\n")
        data = [
            line.split(",")[:-1]
            for i, line in enumerate(file)
            if len(line.split(",")) == num_metrics + 1
        ]
        for i, list_sub in enumerate(data):
            if len(list_sub) < 2:
                continue
            list1 = []
            for j, subst in enumerate(list_sub):
                if len(subst) < 2:
                    continue
                key, value = subst.split(":")
                # remove space in string
                key = key.strip()
                list1.append(round(float(value), 2))
                if i == 0:
                    list_keys.append(key)
            list_values.append(list1)

        for i, key in enumerate(list_keys):
            dic_data[key] = np.array(list_values).T[:][i]

        return dic_data


def make_pretty_plot(ax):
    # use a gray background
    ax.set_facecolor("#E6E6E6")
    ax.set_axisbelow(True)

    # draw solid white grid lines
    plt.grid(color="w", linestyle="solid")

    # hide axis spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # hide top and right ticks
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    # lighten ticks and labels
    ax.tick_params(colors="gray", direction="out")
    for tick in ax.get_xticklabels():
        tick.set_color("gray")
    for tick in ax.get_yticklabels():
        tick.set_color("gray")


def show_train_val_results(keys, values, save_file_name):
    ind = np.arange(int(len(keys) / 2))
    fig, ax = plt.subplots()

    make_pretty_plot(ax)

    width = 0.3
    c1 = "xkcd:salmon"
    c2 = "#0e9aa7"
    bar1 = ax.bar(
        ind,
        values[0:4],
        width,
        label="Train metrics",
        color=[c1, c1, c1, c1],
        align="center",
        alpha=0.5,
    )
    ax.bar_label(bar1, padding=3)
    bar2 = ax.bar(
        ind + width,
        values[4:],
        width,
        label="Validation metrics",
        color=[c2, c2, c2, c2],
        align="center",
        alpha=0.5,
    )
    ax.bar_label(bar2, padding=3)
    plt.xticks(
        ind + width / 2, ("Acuracy", "Fscore", "Precision", "Recall"), rotation=0
    )
    plt.legend(loc="lower center")
    plt.tight_layout()
    fig.savefig(fname=save_file_name, format="pdf", bbox_inches="tight")
    plt.show()


def plot_metrics_results(path_files):
    # colors = ['red', 'orange', 'gold', 'lawngreen', 'lightseagreen', 'royalblue', 'blueviolet',
    #             'red', 'orange', 'gold', 'lawngreen', 'lightseagreen', 'royalblue', 'blueviolet',
    #             'orange', 'gold']

    colors = ["gold", "lawngreen", "royalblue"]

    for path in path_files:
        if os.path.exists(path):
            model_name = path.split("\\")[-1].split(".")[0].split("_")[0]
            print("model name: ", model_name)

            dict_data = arrange_metrics_kfold(path)
            keys = list(dict_data.keys())
            values = list(dict_data.values())

            # index_show = (14,15,16,17,6,7,8,9)
            index_show = (14, 15, 16, 17, 6, 7, 8, 9)
            keys = [keys[i] for i in index_show]
            values = [values[i] for i in index_show]
            max_vals = [round(val.max(), 2) for val in values]
            mean_vals = [round(val.mean(), 2) for val in values]

            # [print(f'men of {key}: {mean_vals[i]}') for i,key in enumerate(keys)]
            [print(f"max of {key}: {max_vals[i]}") for i, key in enumerate(keys)]

            save_file_name = "train_val_metrics_plot.pdf"
            show_train_val_results(keys, max_vals, save_file_name)

        else:
            raise Exception("the path of file is not correct.")


def get_train_val_fscore(path_file):
    if os.path.exists(path_file):
        model_name = path_file.split("\\")[-1].split(".")[0].split("_")[0]
        # print("model name: ", model_name)
        dict_data = arrange_metrics_kfold(path_file)
        keys = list(dict_data.keys())
        values = list(dict_data.values())
        # index_show = (14,15,16,17,6,7,8,9)
        # keys = [keys[i] for i in index_show]
        # values = [values[i] for i in index_show]
        max_vals = [round(val.max(), 2) for val in values]
        mean_vals = [round(val.mean(), 2) for val in values]

        # [print(f'max of {key}: {max_vals[i]}') for i,key in enumerate(keys)]
        vals = [
            max_vals[i]
            for i, key in enumerate(keys)
            if key == "average_train_f1" or key == "average_test_f1"
        ]
        avg_train_f1, avg_val_f1 = vals
        return avg_train_f1, avg_val_f1
    else:
        raise Exception("the path of file is not correct.")


def plot_methods_feat_res(
    methods_train_res,
    methods_test_res,
    save_path,
    methods_features_size,
    models_name,
    methods_name,
):
    fig1 = plt.figure(figsize=(8, 8))
    fig1.suptitle(
        f"Comparison of detection and tracking train results (algorithms subgroups)"
    )
    for k, model in enumerate(models_name):
        zip_train_track = zip(
            track_methods_train_res[model][methods_name[0]],
            track_methods_train_res[model][methods_name[1]],
            track_methods_train_res[model][methods_name[2]],
            track_methods_train_res[model][methods_name[3]],
            track_methods_train_res[model][methods_name[4]],
            track_methods_train_res[model][methods_name[5]],
            track_methods_train_res[model][methods_name[6]],
            track_methods_train_res[model][methods_name[7]],
            track_methods_train_res[model][methods_name[8]],
            track_methods_train_res[model][methods_name[9]],
        )

        zip_train_det = zip(
            det_methods_train_res[model][methods_name[0]],
            det_methods_train_res[model][methods_name[1]],
            det_methods_train_res[model][methods_name[2]],
            det_methods_train_res[model][methods_name[3]],
            det_methods_train_res[model][methods_name[4]],
            det_methods_train_res[model][methods_name[5]],
            det_methods_train_res[model][methods_name[6]],
            det_methods_train_res[model][methods_name[7]],
            det_methods_train_res[model][methods_name[8]],
            det_methods_train_res[model][methods_name[9]],
        )

        det_mean_methods_train_res = [float(sum(val) / 10) for val in zip_train_det]

        tracK_mean_methods_train_res = [float(sum(val) / 10) for val in zip_train_track]

        methods_train_res = [det_mean_methods_train_res, tracK_mean_methods_train_res]

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        plt.subplot(3, 1, k + 1)
        plt.plot(
            methods_features_size,
            methods_train_res[0],
            color=colors[0],
            label="detection",
        )
        plt.plot(
            methods_features_size,
            methods_train_res[1],
            color=colors[1],
            label="tracking",
        )
        plt.xlabel("Feature size")
        if k == 1:
            plt.ylabel("Average Fscore of ten algorithms")
        plt.title(f"{model} test results")
    plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, -1), fancybox=True, shadow=True, ncol=2
    )
    plt.tight_layout()
    fig1.savefig(
        f"{save_path}/train_comp_methods_res.pdf", format="pdf", bbox_inches="tight"
    )
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    fig2 = plt.figure(figsize=(8, 8))
    fig2.suptitle(
        f"Comparison of detection and tracking test results (algorithms subgroups)"
    )
    for k, model in enumerate(models_name):

        zip_test_track = zip(
            track_methods_test_res[model][methods_name[0]],
            track_methods_test_res[model][methods_name[1]],
            track_methods_test_res[model][methods_name[2]],
            track_methods_test_res[model][methods_name[3]],
            track_methods_test_res[model][methods_name[4]],
            track_methods_test_res[model][methods_name[5]],
            track_methods_test_res[model][methods_name[6]],
            track_methods_test_res[model][methods_name[7]],
            track_methods_test_res[model][methods_name[8]],
            track_methods_test_res[model][methods_name[9]],
        )

        zip_test_det = zip(
            det_methods_test_res[model][methods_name[0]],
            det_methods_test_res[model][methods_name[1]],
            det_methods_test_res[model][methods_name[2]],
            det_methods_test_res[model][methods_name[3]],
            det_methods_test_res[model][methods_name[4]],
            det_methods_test_res[model][methods_name[5]],
            det_methods_test_res[model][methods_name[6]],
            det_methods_test_res[model][methods_name[7]],
            det_methods_test_res[model][methods_name[8]],
            det_methods_test_res[model][methods_name[9]],
        )

        det_mean_methods_test_res = [float(sum(val) / 10) for val in zip_test_det]

        tracK_mean_methods_test_res = [float(sum(val) / 10) for val in zip_test_track]

        methods_test_res = [det_mean_methods_test_res, tracK_mean_methods_test_res]

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        plt.subplot(3, 1, k + 1)
        plt.plot(
            methods_features_size,
            methods_test_res[0],
            color=colors[0],
            label="detection",
        )
        plt.plot(
            methods_features_size,
            methods_test_res[1],
            color=colors[1],
            label="tracking",
        )
        plt.xlabel("Feature size")
        if k == 1:
            plt.ylabel("Average Fscore of ten algorithms")
        plt.title(f"{model} test results")
    plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, -1), fancybox=True, shadow=True, ncol=2
    )
    plt.tight_layout()
    fig2.savefig(
        f"{save_path}/test_comp_methods_res.pdf", format="pdf", bbox_inches="tight"
    )
    plt.show(block=False)
    plt.pause(1)
    plt.close()


def plot_manual_feat_res(
    manual_train_res, manual_test_res, save_path, manual_features_size, models_name
):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig1 = plt.figure(figsize=(8, 8))
    fig1.suptitle(f"Comparison of detection and tracking results (manual subgruops)")
    for k, model in enumerate(models_name):
        plt.subplot(3, 1, k + 1)
        plt.plot(
            manual_features_size,
            manual_train_res[0][str(model)]["manual"],
            color=colors[0],
            label="detection",
        )
        plt.plot(
            manual_features_size,
            manual_train_res[1][str(model)]["manual"],
            color=colors[1],
            label="tracking",
        )
        plt.xlabel("Feature size")
        plt.ylabel("Average Fscore")
        plt.title(f"{model} train results")
        plt.tight_layout()
    plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, -1), fancybox=True, shadow=True, ncol=2
    )
    plt.tight_layout()
    fig1.savefig(
        f"{save_path}/train_comp_manual_res.pdf", format="pdf", bbox_inches="tight"
    )
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    fig2 = plt.figure(figsize=(8, 8))
    fig2.suptitle(f"Comparison of detection and tracking results (manual subgruops)")
    for k, model in enumerate(models_name):
        plt.subplot(3, 1, k + 1)
        plt.plot(
            manual_features_size,
            manual_test_res[0][str(model)]["manual"],
            color=colors[0],
            label="detection",
        )
        plt.plot(
            manual_features_size,
            manual_test_res[1][str(model)]["manual"],
            color=colors[1],
            label="tracking",
        )
        plt.xlabel("Feature size")
        plt.ylabel("Average Fscore")
        plt.title(f"{model} test results")
        plt.tight_layout()
    plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, -1), fancybox=True, shadow=True, ncol=2
    )
    plt.tight_layout()
    fig2.savefig(
        f"{save_path}/test_comp_manual_res.pdf", format="pdf", bbox_inches="tight"
    )
    plt.show(block=False)
    plt.pause(1)
    plt.close()


def get_features_results(
    path_folders,
    save_path,
    mode,
    models_name,
    methods_name,
    manual_features_size,
    methods_features_size,
):

    methods_train_res = {
        "gru": {
            "ANOVA": [],
            "chi2": [],
            "kruskal": [],
            "cmim": [],
            "disr": [],
            "mifs": [],
            "ReliefF": [],
            "SURF": [],
            "SURFstar": [],
            "MultiSURF": [],
        },
        "lstm": {
            "ANOVA": [],
            "chi2": [],
            "kruskal": [],
            "cmim": [],
            "disr": [],
            "mifs": [],
            "ReliefF": [],
            "SURF": [],
            "SURFstar": [],
            "MultiSURF": [],
        },
        "mlp": {
            "ANOVA": [],
            "chi2": [],
            "kruskal": [],
            "cmim": [],
            "disr": [],
            "mifs": [],
            "ReliefF": [],
            "SURF": [],
            "SURFstar": [],
            "MultiSURF": [],
        },
    }
    methods_val_res = {
        "gru": {
            "ANOVA": [],
            "chi2": [],
            "kruskal": [],
            "cmim": [],
            "disr": [],
            "mifs": [],
            "ReliefF": [],
            "SURF": [],
            "SURFstar": [],
            "MultiSURF": [],
        },
        "lstm": {
            "ANOVA": [],
            "chi2": [],
            "kruskal": [],
            "cmim": [],
            "disr": [],
            "mifs": [],
            "ReliefF": [],
            "SURF": [],
            "SURFstar": [],
            "MultiSURF": [],
        },
        "mlp": {
            "ANOVA": [],
            "chi2": [],
            "kruskal": [],
            "cmim": [],
            "disr": [],
            "mifs": [],
            "ReliefF": [],
            "SURF": [],
            "SURFstar": [],
            "MultiSURF": [],
        },
    }

    manual_train_res = {
        "gru": {"manual": []},
        "lstm": {"manual": []},
        "mlp": {"manual": []},
    }
    manual_val_res = {
        "gru": {"manual": []},
        "lstm": {"manual": []},
        "mlp": {"manual": []},
    }
    folders_path = [
        os.path.join(path_folders, name) for name in os.listdir(path_folders)
    ]
    for folder_path in folders_path:
        base_name = os.path.basename(folder_path)
        # print(base_name)
        splited_name = base_name.split("_")
        print(splited_name)
        model_name = splited_name[0]
        if len(splited_name) > 2 and not splited_name[1].isdigit():
            if str(splited_name[2]) not in methods_features_size:
                continue
            num_features = splited_name[2]
            method_name = splited_name[1]
        else:
            if str(splited_name[1]) not in [
                (val.split("_")[0]) for val in manual_features_size
            ]:
                continue
            num_features = splited_name[1]
            method_name = "manual"
        file_name = [f for f in os.listdir(folder_path) if "check.txt" in f.split("_")][
            0
        ]
        path_file = os.path.join(folder_path, file_name)
        assert os.path.exists(path_file), f"file {path_file} is not exist"
        # print(path_file)
        avg_train_f1, avg_test_f1 = get_train_val_fscore(path_file)
        # print(f'model_name: {model_name}, method_name: {method_name}, num_features: {num_features}, train_f1: {avg_train_f1}, val_f1: {avg_test_f1}')
        if method_name == "manual":
            print(
                f"model_name: {model_name}, method_name: {method_name}, num_features: {num_features}, train_f1: {avg_train_f1}, val_f1: {avg_test_f1}"
            )
            manual_train_res[model_name][method_name].append(avg_train_f1)
            manual_val_res[model_name][method_name].append(avg_test_f1)
        else:
            methods_train_res[model_name][method_name].append(avg_train_f1)
            methods_val_res[model_name][method_name].append(avg_test_f1)

    print(len(methods_train_res["lstm"]["SURF"]), len(methods_val_res["lstm"]["SURF"]))
    print(
        len(manual_train_res["lstm"]["manual"]), len(manual_val_res["lstm"]["manual"])
    )
    print(manual_train_res["lstm"]["manual"])

    return manual_train_res, manual_val_res, methods_train_res, methods_val_res


if __name__ == "__main__":

    models_name = ["gru", "lstm", "mlp"]
    methods_name = [
        "ANOVA",
        "chi2",
        "cmim",
        "disr",
        "kruskal",
        "mifs",
        "MultiSURF",
        "ReliefF",
        "SURFstar",
        "SURF",
    ]
    manual_features_size = ["34", "36", "58", "62_rba", "62_rsa", "62_rsd", "70"]
    methods_features_size = [
        "20",
        "30",
        "40",
        "50",
        "60",
    ]  # ['10', '20', '30', '40', '50', '60']

    modes = ["detection", "tracking"]

    save_path = f"results/2/comp"
    os.makedirs(save_path, exist_ok=True)

    path_det_folders = f"results/2/200/detection"
    path_track_folders = f"results/2/200/tracking"
    path_folders = [path_det_folders, path_track_folders]

    (
        det_manual_train_res,
        det_manual_test_res,
        det_methods_train_res,
        det_methods_test_res,
    ) = get_features_results(
        path_folders[0],
        save_path,
        modes[0],
        models_name,
        methods_name,
        manual_features_size,
        methods_features_size,
    )

    (
        track_manual_train_res,
        track_manual_test_res,
        track_methods_train_res,
        track_methods_test_res,
    ) = get_features_results(
        path_folders[1],
        save_path,
        modes[1],
        models_name,
        methods_name,
        manual_features_size,
        methods_features_size,
    )

    manual_train_res = [det_manual_train_res, track_manual_train_res]
    manual_test_res = [det_manual_test_res, track_manual_test_res]

    plot_manual_feat_res(
        manual_train_res, manual_test_res, save_path, manual_features_size, models_name
    )

    plot_methods_feat_res(
        manual_train_res,
        manual_test_res,
        save_path,
        methods_features_size,
        models_name,
        methods_name,
    )
