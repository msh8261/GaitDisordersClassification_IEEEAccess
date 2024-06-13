# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103, e1101
import os

import cv2
import numpy as np
import torch

import config.config_data as config
import src.augmentation as aug
# import src.feature_selection as fs
import src.feature_selection as fs
import src.image_filters as fl
from prepare_dataset.manual_feat.utils_data import *
from prepare_dataset.tools.dataReader import DataReader
from prepare_dataset.tools.decorators import (countcall, dataclass, logger,
                                              timeit)
from prepare_dataset.tools.detector import SkeletonDetector
from prepare_dataset.tools.imageReader import ImageReader
from prepare_dataset.tools.tracker import SkeletonTracker
from src.draw_skeleton import draw_skeleton_per_person

# import glob


@logger
def get_keypoints_features_arr(dtime, norm_arr_current):
    """extract key points features"""
    p_arr = [item for sublist in norm_arr_current for item in sublist]
    if len(p_arr) != 34:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("[INFO] len of all features must be 34 in mode is set.")
        print(f"len of feautres: {len(p_arr)}")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    p_arr = [dtime] + p_arr
    p_list_to_file = prepare_list_to_save_in_file(p_arr)
    return p_list_to_file


@logger
def get_features_from_image(images_path, skd, skt, device, mode, show_fig=False):
    """apply keypoint detector on image ket skeleton data and apply defined feature selecion."""
    last_missing_keypoint_kalman = torch.unsqueeze(
        torch.tensor(np.random.random_sample(size=(num_keypoints, 2))), dim=0
    ).to(device)
    last_frame_keypoint_kalman = torch.unsqueeze(
        torch.tensor(np.random.random_sample(size=(num_keypoints, 2))), dim=0
    ).to(device)
    cnt = 0
    ct = 0
    last_time = 0
    first_detection = True
    norm_arr_last = np.zeros((num_keypoints, 2))
    features_vecs = []
    ims = []
    ims_path = []
    for image_path in images_path:
        cnt += 1
        real_time_frame = find_time(image_path)

        ir = ImageReader(image_path)

        img0 = ir.get_image()
        # h, w = img0.shape[:2]

        img1 = img0.copy()

        filtered_image_mix = fl.apply_hist_colormap_filter(img1)
        filtered_image_hist = fl.apply_equalhist_filter(img1)

        persons_det, p_inds, keypoints_scores, im0 = (
            skd.select_best_detection_filter_result(
                filtered_image_mix, filtered_image_hist
            )
        )

        if mode == "1" and len(keypoints_scores) > 0:
            img1 = im0.copy()
            img_det = skd.get_image_with_keypoints(img1, persons_det)
            persons = persons_det
            img_ = img_det
            im = im0.copy()
        elif mode == "2" and len(keypoints_scores) > 0:
            img2 = im0.copy()
            persons_miss_tr = skt.get_tracked_missing_points(persons_det)
            img_miss_points_tr = skd.get_image_with_keypoints(img2, persons_miss_tr)
            persons = persons_miss_tr
            img_ = img_miss_points_tr
            im = im0.copy()
        elif mode == "3":
            img3 = im0.copy()
            persons_miss_det_tr = skt.get_tracked_one_frame_missing_detection(
                persons_det, last_missing_keypoint_kalman
            )
            img_miss_det_tr = skd.get_image_with_keypoints(img3, persons_miss_det_tr)
            last_missing_keypoint_kalman = persons_miss_det_tr
            persons = persons_miss_det_tr
            img_ = img_miss_det_tr
            im = im0.copy()
        elif mode == "4":
            img4 = im0.copy()
            persons_frame_tr = skt.get_tracked_five_frames(
                persons_det, last_frame_keypoint_kalman, cnt
            )
            img_frame_tr = skd.get_image_with_keypoints(img4, persons_frame_tr)
            last_frame_keypoint_kalman = persons_frame_tr
            persons = persons_frame_tr
            img_ = img_frame_tr
            im = im0.copy()
        else:
            print("INFO: please correct the mode of detection or tracking.")

        if len(persons) == 1 and ct < WINDOW_SIZE:
            im_path = image_path

            h, w = im.shape[:2]

            flag, points_arr = fs.check_to_get_all_features_available_in_image(
                h, w, persons
            )

            current_time = real_time_frame

            if bool(flag):
                dtime = get_dtime(current_time, last_time)
                if first_detection:
                    dtime = 0

                first_detection = False

                last_time = current_time

                norm_arr_current = fs.normalize_values_from_image(points_arr, h, w)

                features_arr = get_keypoints_features_arr(dtime, norm_arr_current)

                ct += 1

                # print(features_arr)
                print(len(features_arr.split(",")))
                print(max(features_arr.split(",")))
                print(min(features_arr.split(",")))

                features_vecs.append(features_arr)
                ims.append(im)
                ims_path.append(im_path)

        if show_fig:
            cv2.imshow("win", img_)
            key = cv2.waitKey(1)
            if key == 27:
                break

    return features_vecs, ims, ims_path, ct


@logger
def write_features_on_file(features_vecs, output_data_train, ct):
    """write features and add zero padding in lines to the file"""
    with open(output_data_train, "w", encoding="utf8") as fx:
        for features_arr in features_vecs:
            fx.writelines(features_arr)
            fx.write("\n")
        collect = 0
        if 0 < ct < WINDOW_SIZE:
            collect = ct
            for i in range(WINDOW_SIZE - ct):
                ct += 1
                if ZERO_PADDING:
                    padding = add_zero_padding(num_keypoints)
                    fx.writelines(padding)
                else:
                    fx.writelines(features_arr)
                fx.write("\n")

            return collect
        elif ct == 0:
            return 0


@logger
def write_selected_images(imgs, imgs_path, save_path):
    """write and save selected image used by detector"""
    for i, img_path in enumerate(imgs_path):
        filename = os.path.basename(img_path)
        # cv2.imencode(".jpg", imgs[i])[1].tofile(os.path.join(save_path, filename))
        cv2.imwrite(os.path.join(save_path, filename), imgs[i])


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
def stacking_files(base_dst, folder_patient, folders_date):
    folder_patient_id = os.path.basename(folder_patient)
    folders_date = [os.path.basename(f) for f in folders_date]
    for folder_date in folders_date:
        base_dir = os.path.join(base_dst, folder_patient_id, folder_date)
        data_name = folder_patient_id + "_" + folder_date + "_stacked.File"
        dst_data_path = os.path.join(
            base_dst, folder_patient_id, folder_date, data_name
        )
        data_files_path = [
            os.path.join(base_dir, name)
            for name in os.listdir(base_dir)
            if name.endswith(".File")
        ]
        merge_and_save_files(data_files_path, dst_data_path)
        print(f"INFO: Stacked files for {folder_date} is written...")


@countcall
@timeit
@logger
def main(dir_patients, dir_dist, pm):
    dr = DataReader(dir_patients)
    skd = SkeletonDetector()
    # model = skd.get_detector_model()
    device = skd.get_device()
    skt = SkeletonTracker(device)
    ids, labels = get_labels_ids_from_csv(path_csv_file)
    list_low_detection = []
    list_failed_folders = []
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
                for folder_exercise in dr.get_folders_path(folder_date):
                    dir_dist_3 = get_dist_path(dir_dist_2, folder_exercise)
                    for folder_img in dr.get_folders_path(folder_exercise):
                        images_path = dr.get_files_path(folder_img, format_file="jpg")
                        dir_dist_4 = get_dist_path(dir_dist_3, folder_img)

                        feats_vecs, imgs_vecs, imgs_path, ct = get_features_from_image(
                            images_path, skd, skt, device, pm["mode"], pm["show_fig"]
                        )

                        if pm["write_to_file"]:
                            data_file_name = (
                                os.path.basename(folder_patient)
                                + "_"
                                + os.path.basename(folder_date)
                                + "_"
                                + os.path.basename(folder_exercise)
                                + "_"
                                + data_file_name_
                            )
                            output_data_train = os.path.join(dir_dist_2, data_file_name)
                            flag = write_features_on_file(
                                feats_vecs, output_data_train, ct
                            )

                            save_imgs_path = dir_dist_4
                            write_selected_images(imgs_vecs, imgs_path, save_imgs_path)

                            """save folder name of less detection"""
                            if flag == 0:
                                info2 = folder_exercise
                                info = info2 + ":" + str(flag)
                                print("list_failed_folders: ", info)
                                list_failed_folders.append(info)
                            elif flag != None and flag != 0:
                                info1 = folder_exercise
                                info = info1 + ":" + str(flag)
                                print("list_low_detection: ", info)
                                list_low_detection.append(info)
                            print(f"Data for {folder_exercise} is written...")

            if pm["stacking_three_exercises"]:
                # you can do stacking here
                stacking_files(
                    dir_dist, folder_patient, dr.get_folders_path(folder_patient)
                )

    if pm["save_folders_name_with_less_detection"]:
        path1 = "results/list_low_detection_final.txt"
        path2 = "results/list_failed_folders_final.txt"
        write_lists_to_files(list_low_detection, list_failed_folders, path1, path2)


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

    print(dir_patients)
    print(dir_dist)

    is_not_test = True

    params = dict(
        # '1': detection, '2': detection with missing points
        # '3': one frame tracking, '4': five frames tracking
        mode=mode,
        # to show keypoints on image
        show_fig=False,
        # write selected features into file
        write_to_file=is_not_test,
        # stacking three exercises into one file
        stacking_three_exercises=is_not_test,
        # get the name of folders which had lower detection than threshold window
        save_folders_name_with_less_detection=is_not_test,
    )

    main(dir_patients, dir_dist, params)
