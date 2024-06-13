import os
import sys

# import cv2
import matplotlib.pyplot as plt
import numpy as np

# import random


sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")


def get_labels_ids_from_csv(path_csv_file):
    with open(path_csv_file, "r") as file:
        list_of_ids = []
        list_of_labels = []
        lines = file.readlines()[1:]
        for line in lines:  # read rest of lines
            line = [int(x) for x in line.split(",")]
            list_of_ids.append(line[0])
            list_of_labels.append(line[1])

    return list_of_ids, list_of_labels


def mean(features, ind, keypoints):
    return sum([f[ind] for f in features]) / len(keypoints)


def std(features, ind, keypoints):
    return np.sqrt(
        np.sum(
            [np.power((f[ind] - mean(features, ind, keypoints)), 2) for f in features]
        )
        / len(keypoints)
    )


def analysis_on_features(arr, id, label, w, h):
    xs = []
    ys = []
    ds = []
    angs = []
    lbs = []

    for i in range(len(arr)):
        features = arr[i]

        features = features.split(",")
        features = [(float(j)) for j in features]

        features1 = features[: (keypoints) * 2]
        blocks1 = int(len(features1) / 2)
        features2 = features[(keypoints) * 2 : (keypoints) * 2 + 16]
        blocks2 = int(len(features2) / 2)
        features3 = features[(keypoints) * 2 + 16 : (keypoints) * 2 + 24]
        blocks3 = int(len(features3) / 1)
        features4 = features[(keypoints) * 2 + 24 :]
        blocks4 = int(len(features4) / 1)
        features1 = np.array(np.split(np.array(features1), blocks1))
        features2 = np.array(np.split(np.array(features2), blocks2))
        features3 = np.array(np.split(np.array(features3), blocks3))
        features4 = np.array(np.split(np.array(features4), blocks4))
        x = np.array([fea[0] for fea in features1])
        y = np.array([fea[1] for fea in features1])
        d = np.array([fea for fea in features4])
        ang2 = np.array([fea for fea in features3])

        xs.append(x.mean())
        ys.append(y.mean())
        ds.append(d.mean())
        angs.append(ang2.mean())
        lbs.append(label)

        # plt.plot(x)
        # plt.plot(y)
        # plt.show()

        # n, bins, patches = plt.hist(x, label="x")
        # plt.show()
        # n, bins, patches = plt.hist(y, label="y")
        # plt.show()

    # if show_each_fig:
    #     fig, axs = plt.subplots(2, 2, figsize=(8,6))
    #     axs[0][0].plot(xs, label="x")
    #     axs[0][0].legend()
    #     axs[0][1].plot(ys, label="y")
    #     axs[0][1].legend()
    #     axs[1][0].plot(angs, label="angle2")
    #     axs[1][0].legend()
    #     axs[1][1].plot(ds, label="distance")
    #     axs[1][1].legend()
    #     plt.show()

    if show_each_fig:
        N = len(xs)
        alpha = 0.2
        fig, axs = plt.subplots(3, 2, figsize=(8, 6))
        axs[0][0].scatter(xs, ys, label="x,y", c=np.random.rand(N), alpha=alpha)
        axs[0][0].legend()
        axs[0][1].scatter(xs, ds, label="x,d", c=np.random.rand(N), alpha=alpha)
        axs[0][1].legend()
        axs[1][0].scatter(xs, angs, label="x,angle", c=np.random.rand(N), alpha=alpha)
        axs[1][0].legend()
        axs[1][1].scatter(ys, ds, label="y,d", c=np.random.rand(N), alpha=alpha)
        axs[1][1].legend()
        axs[2][0].scatter(ys, angs, label="y,angle", c=np.random.rand(N), alpha=alpha)
        axs[2][0].legend()
        axs[2][1].scatter(ds, angs, label="d,angle", c=np.random.rand(N), alpha=alpha)
        axs[2][1].legend()
        fig.suptitle(f"results for label {lbs[0]}")
        plt.show()

    return xs, ys, d, angs


def analysis(target_label):
    dir_data = dir_patients

    ids, labels = get_labels_ids_from_csv(path_csv_file)

    base_folders = [
        name
        for name in os.listdir(dir_patients)
        if os.path.isdir(os.path.join(dir_patients, name))
    ]

    list_id, list_label = [], []
    list_mean_x, list_mean_y, list_mean_sp, list_mean_ang = [], [], [], []
    list_std_x, list_std_y, list_std_sp, list_std_ang = [], [], [], []

    individuals = [[], [], [], []]

    for folder_patient_id in base_folders:
        id = int(folder_patient_id.lstrip("0"))
        ind = ids.index(id)
        label = labels[ind]
        print("====================================")
        print("lb: ", label)
        print("====================================")
        target_label = str(label)
        if target_label == str(label):
            base_dir = os.path.join(dir_patients, folder_patient_id)
            print(base_dir)
            folders_patient_date_train = [
                name
                for name in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, name))
            ]
            print(folders_patient_date_train)
            for folder_date in folders_patient_date_train:
                base_dir = os.path.join(dir_patients, folder_patient_id, folder_date)
                base_dir = os.path.join(dir_patients, folder_patient_id, folder_date)

                data_file_name = (
                    folder_patient_id + "_" + folder_date + "_data_train.File"
                )

                # print(os.path.join(dir_data, folder_patient_id, data_file_name))
                if not os.path.exists(
                    os.path.join(dir_data, folder_patient_id, data_file_name)
                ):
                    continue
                list_id.append(id)
                list_label.append(label)
                data_file = open(
                    os.path.join(dir_data, folder_patient_id, data_file_name), "r"
                )
                arr = data_file.readlines()

                # img = cv2.imread(final_folder_images[0])
                # h, w = img.shape[:2]
                h, w = 1, 1
                xs, ys, ds, angs2 = analysis_on_features(arr, id, label, w, h)

                individuals[0].append(np.mean(xs))
                individuals[1].append(np.mean(ys))
                individuals[2].append(np.mean(ds))
                individuals[3].append(np.mean(angs2))

                print(f"Test on {folder_date} is done...")

                data_file.close()

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    n, bins, patches = axs[0][0].hist(
        individuals[0], color="r", label="mean of x values in all sessions"
    )
    axs[0][0].legend(loc="upper left")
    n, bins, patches = axs[0][1].hist(
        individuals[1], color="b", label="mean of y values in all sessions"
    )
    axs[0][1].legend(loc="upper left")
    n, bins, patches = axs[1][0].hist(
        individuals[2], color="c", label="mean of distances in all sessions"
    )
    axs[1][0].legend(loc="upper left")
    n, bins, patches = axs[1][1].hist(
        individuals[3], color="k", label="mean of angles in all sessions"
    )
    axs[1][1].legend(loc="upper left")
    # plt.legend(title='Features', title_fontsize=12, loc='center left', bbox_to_anchor=(0.25, 0.85))
    fig.tight_layout()
    fig.savefig("results/figure_cls" + str(target_label), bbox_inches="tight", dpi=600)
    plt.show()

    return (
        list_id,
        list_label,
        list_mean_x,
        list_mean_y,
        list_mean_sp,
        list_mean_ang,
        list_std_x,
        list_std_y,
        list_std_sp,
        list_std_ang,
    )


if __name__ == "__main__":
    # 104 or 70
    original_feats = 70
    # True for best feature selections
    best_feats = True
    # 10 to 70
    input_size = 50

    dir_patients = (
        r"C:\Users\mohsen\Desktop\Postdoc_Upa\Datasets\GaitAnalysis\100_zero_padding"
    )
    path_csv_file = r"C:\Users\mohsen\Desktop\Postdoc_Upa\Datasets\GaitAnalysis\ORL_skeletons_lookup - labels.csv"
    show_each_fig = False

    if best_feats:
        train_dataset_path = f"./data/best_feats/final_{input_size}_best_feats_selections_from_{original_feats}"
        bf_methods = [
            f"ANOVA_{input_size}_selected_best_feats",
            f"chi2_{input_size}_selected_best_feats",
            f"cmim_{input_size}_selected_best_feats",
            f"disr_{input_size}_selected_best_feats",
            f"kruskal_{input_size}_selected_best_feats",
            f"mifs_{input_size}_selected_best_feats",
            f"MultiSURF_{input_size}_selected_best_feats",
            f"ReliefF_{input_size}_selected_best_feats",
            f"SURF_{input_size}_selected_best_feats",
            f"SURFstar_{input_size}_selected_best_feats",
        ]
        bf_method = bf_methods[0]
        X_train_path = os.path.join(train_dataset_path, f"Xtrain_{bf_method}.File")
        y_train_path = os.path.join(train_dataset_path, f"ytrain.File")
    else:
        train_dataset_path = f"./data/{original_feats}_zero_padding_no"
        X_train_path = os.path.join(train_dataset_path, f"Xtrain.File")
        y_train_path = os.path.join(train_dataset_path, f"ytrain.File")

    points_tickness = 2
    points_COLOR = 2
    keypoints = 17

    target_label = "2"
    (
        ids,
        labels,
        x_mean,
        y_mean,
        speed_mean,
        angle_mean,
        x_std,
        y_std,
        speed_std,
        angle_std,
    ) = analysis(target_label)

    # print("total ids: ", len(labels))
    # print("len x_mean: ", len(x_mean))
    # rand_idx = random.sample(range(len(labels)), 20)
    # np.random.shuffle(rand_idx)
    # print("rand_idx: ", rand_idx)

    # fig1 = plt.figure(figsize=(16,8))
    # x_axis = np.arange(len(np.array(labels)[rand_idx]))
    # plt.bar(x_axis +0.20, np.array(x_mean)[rand_idx], width=0.2, label = 'x_mean')
    # plt.bar(x_axis +0.20*2, np.array(y_mean)[rand_idx], width=0.2, label = 'y_mean')
    # plt.bar(x_axis +0.20*3, np.array(speed_mean)[rand_idx], width=0.2, label = 'speed_mean')
    # plt.bar(x_axis +0.20*4, np.array(angle_mean)[rand_idx], width=0.2, label = 'angle_mean')
    # plt.xticks(x_axis,np.array(labels)[rand_idx])
    # plt.legend()
    # plt.show()

    # fig2 = plt.figure(figsize=(16,8))
    # x_axis = np.arange(len(np.array(labels)[rand_idx]))
    # plt.bar(x_axis +0.20, np.array(x_std)[rand_idx], width=0.2, label = 'x_std')
    # plt.bar(x_axis +0.20*2, np.array(y_std)[rand_idx], width=0.2, label = 'y_std')
    # plt.bar(x_axis +0.20*3, np.array(speed_std)[rand_idx], width=0.2, label = 'speed_std')
    # plt.bar(x_axis +0.20*4, np.array(angle_std)[rand_idx], width=0.2, label = 'angle_std')
    # plt.xticks(x_axis,np.array(labels)[rand_idx])
    # plt.legend()
    # plt.show()
