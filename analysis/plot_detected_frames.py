# pylint: disable = too many blank lines, unused variable, c0103, w0621, w0612, e1121
import os

import matplotlib.pyplot as plt

# import numpy as np
# import seaborn as sns
# from pathlib import Path


def get_file_list(path):
    """
    get file list
    return list of file names
    """
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def get_lists_of_values(file):
    """get list of files
    return list of patients and list of number of detected frames
    """
    val_list = []
    patients_list = []
    with open(file, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            patient = line.split(":")[0].split("\\")[0].lstrip("0")
            val = line.split(":")[1].strip()
            val_list.append(float(val))
            patients_list.append(patient)
    return patients_list, val_list


if __name__ == "__main__":
    path = "./results"
    file_list = get_file_list(path)
    print(file_list)
    modes = ["Detection", "Tracking"]
    colors = ["#44a5c2", "#ffae49"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for i, file in enumerate(file_list):
        pat_list, val_list = get_lists_of_values(file)
        # unique_indexes = np.unique(pat_list, return_index=True)
        unique_indexes = [pat_list.index(pat) for pat in (pat_list)]
        print(pat_list)
        print(unique_indexes)
        print(len(pat_list))
        print(len(val_list))
        bar_width = 0.8
        if i == 1:
            bar_width = 0.5
        bars = ax[i].bar(
            pat_list,
            val_list,
            bar_width,
            color=colors[i],
            edgecolor="black",
            linewidth=2,
        )
        ax[i].bar_label(bars, fontsize=12)
        ax[i].xaxis.set_tick_params(labelsize=12)
        ax[i].yaxis.set_tick_params(labelsize=12)
        ax[i].set_xlabel("Patients Number", fontweight="bold", fontsize=12)
        ax[i].set_ylabel(f"Number of {modes[i]} Frames", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{path}/comp_detected_frames.png", dpi=600)
    plt.show()
