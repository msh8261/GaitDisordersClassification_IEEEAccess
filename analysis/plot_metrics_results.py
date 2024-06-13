# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103, e1101
import os

# import pandas as pd
# import seaborn as sn
# import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cycler

# import ast
# import glob


# from itertools import cycle

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


# def plot_methods_feat_res(methods_train_res, methods_val_res, methods_features_size, models_name, methods_name, mode):
#     colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
#     for model in models_name:
#         fig1 = plt.figure(figsize=(8, 6))
#         for i, method in enumerate(methods_name):
#             plt.plot(methods_features_size, methods_train_res[str(model)][method],  color=colors[i], label=method)
#             plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
#             plt.xlabel('Feature size')
#             plt.ylabel('Fscore')
#             plt.title(f'Train results of {model}')
#         plt.tight_layout()
#         fig1.savefig(f'results/{mode}/{model}_{mode}_train_res.pdf', format="pdf", bbox_inches="tight")
#         plt.show(block=False)
#         plt.pause(1)
#         plt.close()
#     for model in models_name:
#         fig2 = plt.figure(figsize=(8, 6))
#         for i, method in enumerate(methods_name):
#             plt.plot(methods_features_size, methods_val_res[str(model)][method], color=colors[i], label=method)
#             plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
#             plt.xlabel('Feature size')
#             plt.ylabel('Fscore')
#             plt.title(f'Test results of {model}')
#         plt.tight_layout()
#         fig2.savefig(f'results/{mode}/{model}_{mode}_test_res.pdf', format="pdf", bbox_inches="tight")
#         plt.show(block=False)
#         plt.pause(1)
#         plt.close()


def plot_methods_feat_res(
    methods_train_res,
    methods_val_res,
    save_path,
    methods_features_size,
    models_name,
    methods_name,
    mode,
):
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
    fig1 = plt.figure(figsize=(8, 12))
    for k, model in enumerate(models_name):
        plt.subplot(3, 1, k + 1)
        for i, method in enumerate(methods_name):
            plt.plot(
                methods_features_size,
                methods_train_res[str(model)][method],
                color=colors[i],
                label=method,
            )
            plt.xlabel("Feature size")
            plt.ylabel("Average Fscore")
        plt.title(f"Train results of {model} ({mode})")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    plt.tight_layout()
    fig1.savefig(f"{save_path}/train_{mode}_res.pdf", format="pdf", bbox_inches="tight")
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    fig2 = plt.figure(figsize=(8, 12))
    for k, model in enumerate(models_name):
        plt.subplot(3, 1, k + 1)
        for i, method in enumerate(methods_name):
            plt.plot(
                methods_features_size,
                methods_val_res[str(model)][method],
                color=colors[i],
                label=method,
            )
            plt.xlabel("Feature size")
            plt.ylabel("Average Fscore")
        plt.title(f"Test results of {model} ({mode})")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    plt.tight_layout()
    fig2.savefig(f"{save_path}/test_{mode}_res.pdf", format="pdf", bbox_inches="tight")
    plt.show(block=False)
    plt.pause(1)
    plt.close()


def plot_manual_feat_res(
    manual_train_res, manual_val_res, save_path, manual_features_size, models_name, mode
):
    fig1 = plt.figure(figsize=(8, 6))
    for k, model in enumerate(models_name):
        plt.plot(
            manual_features_size,
            manual_train_res[str(model)]["manual"],
            label=str(model),
        )
        plt.legend()
        plt.xlabel("Feature size")
        plt.ylabel("Average Fscore")
        plt.title(f"Train results ({mode})")
    plt.tight_layout()
    fig1.savefig(
        f"{save_path}/manual_{mode}_train_res.pdf", format="pdf", bbox_inches="tight"
    )
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    fig2 = plt.figure(figsize=(8, 6))
    for k, model in enumerate(models_name):
        plt.plot(
            manual_features_size, manual_val_res[str(model)]["manual"], label=str(model)
        )
        plt.legend()
        plt.xlabel("Feature size")
        plt.ylabel("Average Fscore")
        plt.title(f"Test results ({mode})")
    plt.tight_layout()
    fig2.savefig(
        f"{save_path}/manual_{mode}_test_res.pdf", format="pdf", bbox_inches="tight"
    )
    plt.show(block=False)
    plt.pause(1)
    plt.close()


def plot_features_resutls(dir, save_path, mode):
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
    folders_path = [os.path.join(dir, name) for name in os.listdir(path_folders)]
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

    plot_manual_feat_res(
        manual_train_res,
        manual_val_res,
        save_path,
        manual_features_size,
        models_name,
        mode,
    )

    plot_methods_feat_res(
        methods_train_res,
        methods_val_res,
        save_path,
        methods_features_size,
        models_name,
        methods_name,
        mode,
    )


if __name__ == "__main__":
    # path_files = glob.glob("./results/comp_new/c3/texts/*.txt")

    # path_files = [r"results\100\detection\dnn_70\dnn_70_fold3_random2_3classes_results_check.txt"]

    # plot_metrics_results(path_files)

    # modes = ['detection', 'tracking' ]
    modes = ["detection"]

    for mode in modes:
        save_path = f"results/2/{mode}"
        os.makedirs(save_path, exist_ok=True)

        dir_path = "results/2/500"
        os.makedirs(dir_path, exist_ok=True)
        path_folders = f"{dir_path}/{mode}"

        plot_features_resutls(path_folders, save_path, mode)
