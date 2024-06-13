# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103
import os
import random

import numpy as np

import src.feature_selection as fs
# import src.augmentation as aug
# import src.feature_selection as fs
# import src.image_filters as fl
# from src.draw_skeleton import draw_skeleton_per_person
# from prepare_dataset.tools.imageReader import ImageReader
# from prepare_dataset.tools.tracker import SkeletonTracker
# from prepare_dataset.tools.detector import SkeletonDetector
from prepare_dataset.manual_feat.utils_data import *
from prepare_dataset.tools.dataReader import DataReader
# import config.config_data as config
from prepare_dataset.tools.decorators import (countcall, dataclass, logger,
                                              timeit)

# import glob
# import cv2
# import time

# import xml.etree.ElementTree as ET
# from xml.etree.ElementTree import Element, SubElement, Comment, tostring
# from xml.etree import ElementTree
# from xml.dom import minidom

# import torch


def add_zero_row(features_size):
    """add zeros to empty rows from windows size"""
    feat = np.zeros(features_size)
    return feat


@logger
def get_arr_with_diff_features_v0(dtime, norm_arr_last, norm_arr_current):
    """
    Get array of normalized keypoints features
    Return array of generatedd features
    """
    if features_size == 84:
        p_arr = fs.add_speed_angle_of_keypoints_in_two_sequences(
            dtime, norm_arr_last, norm_arr_current
        )
        p_arr = fs.add_distance_angle_of_symetric_keypoints_in_a_sequence(
            p_arr, norm_arr_current
        )
        # p_arr.append(dtime)
        if len(p_arr) != 84:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("[INFO] len of all features must be 84 in mode is set.")
            print(f"len of feautres: {len(p_arr)}")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        p_list_to_file = prepare_list_to_save_in_file(p_arr)

    elif features_size == 104:
        # 34 angle and dist need to be add
        p_arr = fs.add_speed_angle_of_keypoints_in_two_sequences(
            dtime, norm_arr_last, norm_arr_current
        )
        # 16 angle and dist need to be add
        p_arr = fs.add_distance_angle_of_symetric_keypoints_in_a_sequence(
            p_arr, norm_arr_current
        )
        # 8 angle need to be add
        p_arr = fs.angles_selected_body_bones(p_arr, norm_arr_current)
        # 12 dist need to be add
        p_arr = fs.add_displacement_pairwise_joints(p_arr, norm_arr_current)
        # p_arr.append(dtime)
        if len(p_arr) != 104:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("[INFO] len of all features must be 104 in mode is set.")
            print(f"len of feautres: {len(p_arr)}")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        p_list_to_file = prepare_list_to_save_in_file(p_arr)

    elif features_size == 68:
        p_arr = fs.add_distance_angle_of_keypoints_in_two_sequences(
            norm_arr_last, norm_arr_current
        )
        if len(p_arr) != 68:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("[INFO] len of all features must be 68 in mode is set.")
            print(f"len of feautres: {len(p_arr)}")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        p_list_to_file = prepare_list_to_save_in_file(p_arr)

    elif features_size == 50:
        # 34 keypoints features array
        p_arr = [item for sublist in norm_arr_current for item in sublist]
        p_arr = fs.add_distance_angle_of_symetric_keypoints_in_a_sequence(
            p_arr, norm_arr_current
        )
        if len(p_arr) != 50:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("[INFO] len of all features must be 50 in mode is set.")
            print(f"len of feautres: {len(p_arr)}")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        p_list_to_file = prepare_list_to_save_in_file(p_arr)

    elif features_size == 34:
        # 34 keypoints features array
        p_arr = [item for sublist in norm_arr_current for item in sublist]
        if len(p_arr) != 34:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("[INFO] len of all features must be 34 in mode is set.")
            print(f"len of feautres: {len(p_arr)}")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        p_list_to_file = prepare_list_to_save_in_file(p_arr)
    else:
        raise ValueError("features_size {features_size} is not supported.")

    return p_list_to_file


@logger
def get_arr_with_70_features(norm_arr_current):
    # 34 keypoints features array
    p_arr = [item for sublist in norm_arr_current for item in sublist]
    # 16 angle and dist need to be add
    p_arr = fs.add_distance_angle_of_symetric_keypoints_in_a_sequence(
        p_arr, norm_arr_current
    )
    # 8 angle need to be add
    p_arr = fs.angles_selected_body_bones(p_arr, norm_arr_current)
    # 12 dist need to be add
    p_arr = fs.add_displacement_pairwise_joints(p_arr, norm_arr_current)
    if len(p_arr) != 70:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("[INFO] len of all features must be 70 in mode is set.")
        print(f"len of feautres: {len(p_arr)}")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    p_list_to_file = prepare_list_to_save_in_file(p_arr)

    return p_list_to_file


@logger
def get_features_arr(norm_arr_current_0):
    """apply all selection methods to extract features for train"""
    # normalized 34 features from skeleton points
    norm_arr_current = np.array_split(norm_arr_current_0, num_keypoints)
    if np.count_nonzero(norm_arr_current_0) == 0:
        p_arr = add_zero_row(features_size)
        p_list_to_file = prepare_list_to_save_in_file(p_arr)
    else:
        p_list_to_file = get_arr_with_70_features(norm_arr_current)

    return p_list_to_file, norm_arr_current


@logger
def get_generated_features(file_path):
    """generate and add features."""
    norm_arr_last = np.zeros((num_keypoints, 2))
    features_vecs = []
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(os.path.basename(file_path))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    with open(file_path, "r", encoding="utf8") as file:
        for line in file:
            # get dtime from array
            dtime = float(line.split(",")[0])
            # remove dtime from array and covert str to float
            norm_arr_current = [float(x) for x in line.split(",")[1:]]

            features_arr_str, norm_arr_last = get_features_arr(norm_arr_current)

            features_vecs.append(features_arr_str)

    return features_vecs


@logger
def write_final_features_on_file(features_vecs, output_data_train):
    """write features and add zero padding in lines to the file"""
    with open(output_data_train, "w") as fx:
        for features_arr in features_vecs:
            fx.writelines(features_arr)
            fx.write("\n")


def get_dist_path(dir_dist, folder_path):
    """make distination path for the file"""
    os.makedirs(os.path.join(dir_dist, os.path.basename(folder_path)), exist_ok=True)
    dir_dist = os.path.join(dir_dist, os.path.basename(folder_path))
    return dir_dist


@logger
def merge_and_save_train_labels(files, dst_label_path):
    # Reading data from file1
    with open(dst_label_path, "w", encoding="utf8") as fw:
        for file in files:
            with open(file, "r", encoding="utf8") as fr:
                data = fr.read()
                fw.write(data)
                fw.write("\n")


@logger
def merge_and_save_test_labels(files, dst_label_path):
    # Reading data from file1
    with open(dst_label_path, "w", encoding="utf8") as fw:
        for file in files:
            with open(file, "r", encoding="utf8") as fr:
                data = fr.read()
                fw.write(data)
                fw.write("\n")


@logger
def split_data_v1(num_subfolders, base_dst, folder_patient, folders_date):
    print("@@@@@@@@@@@@@@@@@@@@@@")
    print(f"{os.path.basename(folder_patient)} has {num_subfolders} subfolders.")
    print("@@@@@@@@@@@@@@@@@@@@@@")
    folder_patient_id = os.path.basename(folder_patient)
    folders_date = [os.path.basename(f) for f in folders_date]
    print(folders_date)
    if num_subfolders > 1:
        random.seed(21)
        folder_date_val = random.choice(folders_date)
        folders_date.remove(folder_date_val)
        base_dir = os.path.join(base_dst, folder_patient_id)
        Label_name = folder_patient_id + "_" + folder_date_val + "_label.File"
        data_name = folder_patient_id + "_" + folder_date_val + "_data_val.File"
        dst_label_path = os.path.join(base_dst, folder_patient_id, Label_name)
        dst_data_path = os.path.join(base_dst, folder_patient_id, data_name)
        data_files_path = [
            os.path.join(base_dir, name)
            for name in os.listdir(base_dir)
            if name.endswith(
                f"{folder_patient_id}_{folder_date_val}_stacked_add_features.File"
            )
        ]
        merge_and_save_files(data_files_path, dst_data_path)
        xml_file = os.path.join(base_dir, folder_date_val, "Notice.xml")
        save_lable_file(xml_file, dst_label_path)
        print(f"INFO: Data val for {folder_date_val} is written...")

    for folder_date in folders_date:
        base_dir = os.path.join(base_dst, folder_patient_id)
        Label_name = folder_patient_id + "_" + folder_date + "_label.File"
        data_name = folder_patient_id + "_" + folder_date + "_data_train.File"
        dst_label_path = os.path.join(base_dst, folder_patient_id, Label_name)
        dst_data_path = os.path.join(base_dst, folder_patient_id, data_name)
        data_files_path = [
            os.path.join(base_dir, name)
            for name in os.listdir(base_dir)
            if name.endswith(
                f"{folder_patient_id}_{folder_date}_stacked_add_features.File"
            )
        ]
        merge_and_save_files(data_files_path, dst_data_path)
        xml_file = os.path.join(base_dir, folder_date, "Notice.xml")
        save_lable_file(xml_file, dst_label_path)
        print(f"INFO: Data train for {folder_date} is written...")


@logger
def make_final_files(dir_dist):
    list_data_split = []
    list_label_split = []
    list_data_val = []
    list_label_val = []
    dr = DataReader(dir_dist)
    for folder_patient in dr.get_folders_path():
        base_dir1 = folder_patient
        folder_patient_id = os.path.basename(folder_patient)
        folders_date = dr.get_folders_path(folder_patient)
        for folder_date in folders_date:
            folder_date = os.path.basename(folder_date)
            data_val_path = (
                base_dir1
                + "/"
                + folder_patient_id
                + "_"
                + folder_date
                + "_data_val.File"
            )
            data_label_path = (
                base_dir1 + "/" + folder_patient_id + "_" + folder_date + "_label.File"
            )
            if os.path.isfile(data_val_path):
                if os.path.isfile(data_label_path):
                    list_data_val.append(data_val_path)
                    list_label_val.append(data_label_path)
                    continue

            data_split_path = (
                base_dir1
                + "/"
                + folder_patient_id
                + "_"
                + folder_date
                + "_data_train.File"
            )
            label_split_path = (
                base_dir1 + "/" + folder_patient_id + "_" + folder_date + "_label.File"
            )

            if os.path.isfile(data_split_path):
                if os.path.isfile(label_split_path):
                    list_data_split.append(data_split_path)
                    list_label_split.append(label_split_path)

    dst_data_train_path = os.path.join(dir_dist, "Xtrain.File")
    dst_label_train_path = os.path.join(dir_dist, "ytrain.File")
    merge_and_save_files(list_data_split, dst_data_train_path)
    merge_and_save_train_labels(list_label_split, dst_label_train_path)

    dst_data_val_path = os.path.join(dir_dist, "Xtest.File")
    dst_label_val_path = os.path.join(dir_dist, "ytest.File")
    merge_and_save_files(list_data_val, dst_data_val_path)
    merge_and_save_test_labels(list_label_val, dst_label_val_path)


@countcall
@timeit
@logger
def main(dir_dist):
    """
    Get directory of data to make file
    """
    dr = DataReader(dir_dist)
    ids, labels = get_labels_ids_from_csv(path_csv_file)
    for folder_patient in dr.get_folders_path():
        dir_dist_1 = get_dist_path(dir_dist, folder_patient)
        id = int(os.path.basename(folder_patient).lstrip("0"))
        if id in ids:
            label = get_label_from_index(labels, ids, id)
            num_subfolders = len(dr.get_folders_path(folder_patient))
            for folder_date in dr.get_folders_path(folder_patient):
                dir_dist_2 = get_dist_path(dir_dist_1, folder_date)
                path_xml_to_save = os.path.join(dir_dist_2, label_file_name_)
                params = get_label_info(folder_patient, folder_date, label)
                prepare_xml_for_labels(params, path_xml_to_save)
                file_name = (
                    os.path.basename(folder_patient)
                    + "_"
                    + os.path.basename(folder_date)
                    + "_stacked.File"
                )
                file_path = os.path.join(dir_dist_2, file_name)
                feats_vecs = get_generated_features(file_path)
                data_file_name = (
                    os.path.basename(folder_patient)
                    + "_"
                    + os.path.basename(folder_date)
                    + "_stacked_add_features.File"
                )
                output_data_train = os.path.join(dir_dist_1, data_file_name)
                write_final_features_on_file(feats_vecs, output_data_train)

            # you can do stacking here
            split_data_v1(
                num_subfolders,
                dir_dist,
                folder_patient,
                dr.get_folders_path(folder_patient),
            )

    make_final_files(dir_dist)


if __name__ == "__main__":
    import yaml

    with open("config/config_data.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # configure the parameters
    dir_patients = config["dir_patients"]
    dir_dist = config["dir_dist"]
    path_csv_file = config["path_csv_file"]

    WINDOW_SIZE = config["dataset_params"]["WINDOW_SIZE"]
    features_size = config["dataset_params"]["features_size"]
    keypoints = config["keypoints"]

    label_file_name_ = config["label_file_name_"]
    data_file_name_ = config["data_file_name_"]

    image_need_crop = config["dataset_params"]["image_need_crop"]
    scale_w = config["dataset_params"]["scale_w"]
    scale_h = config["dataset_params"]["scale_h"]
    ZERO_PADDING = config["dataset_params"]["ZERO_PADDING"]
    person_thresh = config["dataset_params"]["person_thresh"]
    keypoint_threshold = config["dataset_params"]["keypoint_threshold"]
    num_keypoints = config["dataset_params"]["num_keypoints"]
    all_possible_features = config["dataset_params"]["all_possible_features"]
    mode = config["dataset_params"]["mode"]
    if mode == "1" or mode == "2":
        dir_dist = dir_dist + "/detection"
    elif mode == "3" or mode == "4":
        dir_dist = dir_dist + "/tracking"

    if all_possible_features:
        add_speed_angle_features_in_two_sequences = False
        add_distance_angle_features_in_one_sequence = False
        add_distance_angle_features_in_two_sequences = False
    else:
        add_speed_angle_features_in_two_sequences = True
        add_distance_angle_features_in_one_sequence = True
        add_distance_angle_features_in_two_sequences = False

    print(dir_dist)

    main(dir_dist)
