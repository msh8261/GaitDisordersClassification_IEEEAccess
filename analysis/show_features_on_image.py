# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103, e1101, e1124
# import logging
import os
import sys

import cv2
import numpy as np

import src.feature_selection as fs
from nn.tools.keypointrcnn_resnet50_fpn import keypointrcnn_resnet50_fpn
from src.draw_skeleton import draw_skeleton_per_person

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

model, device = keypointrcnn_resnet50_fpn.create()


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def show_points_on_high_quality_image(frame):
    img_src = frame.copy()
    # The variable 'sigma_s' determines how big the neighbourhood of pixels
    # must be to perform filtering.
    # The variable 'sigma_r' determines how the different colours within
    # the neighbourhood of pixels will be averaged with each other.
    # Its range is from: 0 - 1. A smaller value means similar colors
    #  will be averaged out while different colors remain as they are.

    img_res = cv2.detailEnhance(img_src, sigma_s=5, sigma_r=0.15)
    # img_res = super_res(img_res)
    img_tensor = fs.input_for_model(img_res, device)
    output = model(img_tensor)[0]
    skeletal_img = draw_skeleton_per_person(
        img_res,
        output["scores"],
        output["keypoints"],
        output["keypoints_scores"],
        output["scores"],
        keypoint_threshold=2,
    )
    return skeletal_img


def show_points_on_original_image(frame):
    img_src = frame.copy()
    img_tensor = fs.input_for_model(img_src, device)
    output = model(img_tensor)[0]
    skeletal_img = draw_skeleton_per_person(
        img_src,
        output["scores"],
        output["keypoints"],
        output["keypoints_scores"],
        output["scores"],
        keypoint_threshold=2,
    )
    return skeletal_img


def show_line_angle(img, features, line_dist_ang, w, h):
    tk = 1
    color = (50, 250, 50)
    color_text = (255, 50, 50)
    font_family = 1
    font_size = 1
    font_stroke = 1
    i = 1
    cv2.line(
        img,
        [int(features[1][0] * w), int(features[1][1] * h)],
        [int(features[2][0] * w), int(features[2][1] * h)],
        color,
        tk,
    )
    angle = np.round(line_dist_ang[i], 3)
    point = (int(features[1][0] * w), int(features[1][1] * h) + 5)
    cv2.putText(
        img,
        str(angle),
        point,
        font_family,
        font_size,
        color_text,
        font_stroke,
        cv2.LINE_AA,
    )

    cv2.line(
        img,
        [int(features[3][0] * w), int(features[3][1] * h)],
        [int(features[4][0] * w), int(features[4][1] * h)],
        color,
        tk,
    )
    angle = np.round(line_dist_ang[i + 2], 3)
    point = (int(features[3][0] * w), int(features[3][1] * h) + 5)
    cv2.putText(
        img,
        str(angle),
        point,
        font_family,
        font_size,
        color_text,
        font_stroke,
        cv2.LINE_AA,
    )

    cv2.line(
        img,
        [int(features[5][0] * w), int(features[5][1] * h)],
        [int(features[6][0] * w), int(features[6][1] * h)],
        color,
        tk,
    )
    angle = np.round(line_dist_ang[i + 4], 3)
    point = (int(features[5][0] * w), int(features[5][1] * h) + 5)
    cv2.putText(
        img,
        str(angle),
        point,
        font_family,
        font_size,
        color_text,
        font_stroke,
        cv2.LINE_AA,
    )

    cv2.line(
        img,
        [int(features[7][0] * w), int(features[7][1] * h)],
        [int(features[8][0] * w), int(features[8][1] * h)],
        color,
        tk,
    )
    angle = np.round(line_dist_ang[i + 6], 3)
    point = (int(features[7][0] * w), int(features[7][1] * h) + 5)
    cv2.putText(
        img,
        str(angle),
        point,
        font_family,
        font_size,
        color_text,
        font_stroke,
        cv2.LINE_AA,
    )

    cv2.line(
        img,
        [int(features[9][0] * w), int(features[9][1] * h)],
        [int(features[10][0] * w), int(features[10][1] * h)],
        color,
        tk,
    )
    angle = np.round(line_dist_ang[i + 8], 3)
    point = (int(features[9][0] * w), int(features[9][1] * h) + 5)
    cv2.putText(
        img,
        str(angle),
        point,
        font_family,
        font_size,
        color_text,
        font_stroke,
        cv2.LINE_AA,
    )

    cv2.line(
        img,
        [int(features[11][0] * w), int(features[11][1] * h)],
        [int(features[12][0] * w), int(features[12][1] * h)],
        color,
        tk,
    )
    angle = np.round(line_dist_ang[i + 10], 3)
    point = (int(features[11][0] * w), int(features[11][1] * h) + 5)
    cv2.putText(
        img,
        str(angle),
        point,
        font_family,
        font_size,
        color_text,
        font_stroke,
        cv2.LINE_AA,
    )

    cv2.line(
        img,
        [int(features[13][0] * w), int(features[13][1] * h)],
        [int(features[14][0] * w), int(features[14][1] * h)],
        color,
        tk,
    )
    angle = np.round(line_dist_ang[i + 12], 3)
    point = (int(features[13][0] * w), int(features[13][1] * h) + 5)
    cv2.putText(
        img,
        str(angle),
        point,
        font_family,
        font_size,
        color_text,
        font_stroke,
        cv2.LINE_AA,
    )

    cv2.line(
        img,
        [int(features[15][0] * w), int(features[15][1] * h)],
        [int(features[16][0] * w), int(features[16][1] * h)],
        color,
        tk,
    )
    angle = np.round(line_dist_ang[i + 14], 3)
    point = (int(features[15][0] * w), int(features[15][1] * h) + 5)
    cv2.putText(
        img,
        str(angle),
        point,
        font_family,
        font_size,
        color_text,
        font_stroke,
        cv2.LINE_AA,
    )


def show_bones_angle(img, features, bone_ang_dist, w, h):
    tk = 2
    color = (50, 250, 50)
    color_text = (255, 50, 50)
    font_family = 1
    font_size = 1
    font_stroke = 1
    i = 0

    cv2.line(
        img,
        [int(features[5][0] * w), int(features[5][1] * h)],
        [int(features[6][0] * w), int(features[6][1] * h)],
        color,
        tk,
    )
    cv2.line(
        img,
        [int(features[5][0] * w), int(features[5][1] * h)],
        [int(features[7][0] * w), int(features[7][1] * h)],
        color,
        tk,
    )
    angle = np.round(bone_ang_dist[i], 3)
    point = (int(features[5][0] * w), int(features[5][1] * h) + 5)
    cv2.putText(
        img,
        str(angle),
        point,
        font_family,
        font_size,
        color_text,
        font_stroke,
        cv2.LINE_AA,
    )

    cv2.line(
        img,
        [int(features[6][0] * w), int(features[6][1] * h)],
        [int(features[8][0] * w), int(features[8][1] * h)],
        color,
        tk,
    )
    angle = np.round(bone_ang_dist[i + 1], 3)
    point = (int(features[6][0] * w), int(features[6][1] * h) - 5)
    cv2.putText(
        img,
        str(angle),
        point,
        font_family,
        font_size,
        color_text,
        font_stroke,
        cv2.LINE_AA,
    )

    cv2.line(
        img,
        [int(features[7][0] * w), int(features[7][1] * h)],
        [int(features[9][0] * w), int(features[9][1] * h)],
        color,
        tk,
    )
    angle = np.round(bone_ang_dist[i + 2], 3)
    point = (int(features[7][0] * w), int(features[7][1] * h) + 5)
    cv2.putText(
        img,
        str(angle),
        point,
        font_family,
        font_size,
        color_text,
        font_stroke,
        cv2.LINE_AA,
    )

    cv2.line(
        img,
        [int(features[8][0] * w), int(features[8][1] * h)],
        [int(features[10][0] * w), int(features[10][1] * h)],
        color,
        tk,
    )
    angle = np.round(bone_ang_dist[i + 3], 3)
    point = (int(features[8][0] * w), int(features[8][1] * h) - 5)
    cv2.putText(
        img,
        str(angle),
        point,
        font_family,
        font_size,
        color_text,
        font_stroke,
        cv2.LINE_AA,
    )

    cv2.line(
        img,
        [int(features[11][0] * w), int(features[11][1] * h)],
        [int(features[12][0] * w), int(features[12][1] * h)],
        color,
        tk,
    )
    cv2.line(
        img,
        [int(features[11][0] * w), int(features[11][1] * h)],
        [int(features[13][0] * w), int(features[13][1] * h)],
        color,
        tk,
    )
    angle = np.round(bone_ang_dist[i], 3)
    point = (int(features[11][0] * w), int(features[11][1] * h) + 5)
    cv2.putText(
        img,
        str(angle),
        point,
        font_family,
        font_size,
        color_text,
        font_stroke,
        cv2.LINE_AA,
    )

    cv2.line(
        img,
        [int(features[12][0] * w), int(features[12][1] * h)],
        [int(features[14][0] * w), int(features[14][1] * h)],
        color,
        tk,
    )
    angle = np.round(bone_ang_dist[i + 1], 3)
    point = (int(features[12][0] * w), int(features[12][1] * h) - 5)
    cv2.putText(
        img,
        str(angle),
        point,
        font_family,
        font_size,
        color_text,
        font_stroke,
        cv2.LINE_AA,
    )

    cv2.line(
        img,
        [int(features[13][0] * w), int(features[13][1] * h)],
        [int(features[15][0] * w), int(features[15][1] * h)],
        color,
        tk,
    )
    angle = np.round(bone_ang_dist[i + 2], 3)
    point = (int(features[13][0] * w), int(features[13][1] * h) + 5)
    cv2.putText(
        img,
        str(angle),
        point,
        font_family,
        font_size,
        color_text,
        font_stroke,
        cv2.LINE_AA,
    )

    cv2.line(
        img,
        [int(features[14][0] * w), int(features[14][1] * h)],
        [int(features[16][0] * w), int(features[16][1] * h)],
        color,
        tk,
    )
    angle = np.round(bone_ang_dist[i + 3], 3)
    point = (int(features[14][0] * w), int(features[14][1] * h) - 5)
    cv2.putText(
        img,
        str(angle),
        point,
        font_family,
        font_size,
        color_text,
        font_stroke,
        cv2.LINE_AA,
    )


def show_keypoints_features_on_selected_images(final_folder_images, arr):
    for i, img_path in enumerate(final_folder_images):
        frame = cv2.imread(img_path)
        frame_test = frame.copy()
        # frame = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # cv2.normalize(frame, frame_test, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        h, w = frame.shape[:2]
        # make a copy of the frame
        img = frame.copy()

        features0 = arr[i]
        features0 = features0.split(",")
        features0 = [(float(j)) for j in features0[1:]]

        print(f"features size: {len(features0)}")
        print(h, w)

        features = features0[: (keypoints) * 2]
        blocks = int(len(features) / 2)

        # print("blocks: ", blocks)
        features = np.array(np.split(np.array(features), blocks))
        for f in features:
            p = [int(f[0] * w), int(f[1] * h)]
            cv2.circle(img, p, points_tickness, points_COLOR, -1)

        cv2.imshow("frame", img)
        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()


def show_all_features_on_selected_images(images, arr):
    for i, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        frame_test = frame.copy()
        # frame = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # cv2.normalize(frame, frame_test, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        h, w = frame.shape[:2]
        # make a copy of the frame
        img = frame.copy()

        features0 = arr[i]
        features0 = features0.split(",")
        features0 = [(float(j)) for j in features0]

        print(f"features size: {len(features0)}")
        print(h, w)

        features = features0[: (keypoints) * 2]
        blocks = int(len(features) / 2)
        line_dist_ang = features0[(keypoints) * 2 : 50]
        bone_ang = features0[50:58]
        bone_dist = features0[58:]

        # print("blocks: ", blocks)
        features = np.array(np.split(np.array(features), blocks))
        for f in features:
            p = [int(f[0] * w), int(f[1] * h)]
            # print(p)
            cv2.circle(img, p, points_tickness, points_COLOR, -1)

        if len(line_dist_ang) > 1:
            show_line_angle(img, features, line_dist_ang, w, h)
        if len(bone_ang) > 1:
            show_bones_angle(img, features, bone_ang, w, h)
        # cv2.imshow("win", img)
        cv2.imshow("frame", img)

        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()


def show_with_34_feats(dir_patients):
    dir_data = dir_patients
    base_folders = [
        name
        for name in os.listdir(dir_patients)
        if os.path.isdir(os.path.join(dir_patients, name))
    ]
    for folder_patient_id in base_folders:
        base_dir = os.path.join(dir_patients, folder_patient_id)
        folders_patient_date = [
            name
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
        ]
        for folder_date in folders_patient_date:
            base_dir = os.path.join(dir_patients, folder_patient_id, folder_date)
            folders_patient_exercise = [
                name
                for name in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, name))
            ]
            for folder_exercise in folders_patient_exercise:
                base_dir = os.path.join(
                    dir_patients, folder_patient_id, folder_date, folder_exercise
                )
                final_folder = [
                    name
                    for name in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, name))
                ][0]
                final_folder_images = [
                    os.path.join(base_dir, final_folder, name)
                    for name in os.listdir(os.path.join(base_dir, final_folder))
                ]
                data_file_name = (
                    folder_patient_id
                    + "_"
                    + folder_date
                    + "_"
                    + folder_exercise
                    + "_data.File"
                )
                print(data_file_name)
                with open(
                    os.path.join(
                        dir_data, folder_patient_id, folder_date, data_file_name
                    ),
                    "r",
                ) as data_file:
                    arr = data_file.readlines()
                    show_keypoints_features_on_selected_images(final_folder_images, arr)
                    print(f"Test on {folder_exercise} is done...")


def show_with_final_file(dir_patients):
    dir_data = dir_patients
    base_folders = [
        name
        for name in os.listdir(dir_patients)
        if os.path.isdir(os.path.join(dir_patients, name))
    ]
    imgs = []
    for folder_patient_id in base_folders:
        base_dir = os.path.join(dir_patients, folder_patient_id)
        folders_patient_date = [
            name
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
        ]
        folders_patient_date = [
            name
            for name in folders_patient_date
            if not os.path.exists(
                os.path.join(
                    dir_data,
                    folder_patient_id,
                    f"{folder_patient_id}_{name}_data_val.File",
                )
            )
        ]
        for folder_date in folders_patient_date:
            base_dir = os.path.join(dir_patients, folder_patient_id, folder_date)
            folders_patient_exercise = [
                name
                for name in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, name))
            ]
            for folder_exercise in folders_patient_exercise:
                base_dir = os.path.join(
                    dir_patients, folder_patient_id, folder_date, folder_exercise
                )
                final_folder = [
                    name
                    for name in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, name))
                ][0]
                final_folder_images = [
                    os.path.join(base_dir, final_folder, name)
                    for name in os.listdir(os.path.join(base_dir, final_folder))
                ]
                imgs.extend(final_folder_images)
    with open(os.path.join(dir_data, "Xtrain.File"), "r") as data_file:
        arr = data_file.readlines()
        show_all_features_on_selected_images(imgs, arr)
        print(f"Test on {folder_exercise} is done...")


def show_features_on_selected_images(final_folder_images, arr):
    for i, img_path in enumerate(final_folder_images):
        frame = cv2.imread(img_path)
        frame_test = frame.copy()
        # frame = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # cv2.normalize(frame, frame_test, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        h, w = frame.shape[:2]
        # make a copy of the frame
        img = frame.copy()

        # skeletal_img1 = show_points_on_high_quality_image(frame)
        # skeletal_img2 = show_points_on_high_quality_image(frame_test)

        # skeletal_img = show_points_on_high_quality_image(frame)
        # skeletal_img = show_points_on_original_image(frame)

        features0 = arr[i]
        features0 = features0.split(",")
        features0 = [(float(j)) for j in features0]

        print(f"initial features size: {len(features0)}")

        if (
            add_speed_angle_features_in_two_sequences
            and add_distance_angle_features_in_one_sequence
        ):
            features = features0[: (keypoints) * 4]
            blocks = int(len(features) / 4)
            line_dist_ang = features0[(keypoints) * 4 : 84]
            bone_ang_dist = features0[84:]
        elif add_distance_angle_features_in_one_sequence:
            features = features0[: (keypoints) * 2]
            blocks = int(len(features) / 2)
            line_dist_ang = features0[(keypoints) * 4 : 84]
            bone_ang_dist = features0[84:]
        elif add_distance_angle_features_in_two_sequences:
            features = features0[: (keypoints) * 4]
            blocks = int(len(features) / 4)
            line_dist_ang = features0[(keypoints) * 4 : 84]
            bone_ang_dist = features0[84:]
        elif len(features0) == 35:
            features0 = features0[1:]
            features = features0[: (keypoints) * 2]
            blocks = int(len(features) / 2)
            line_dist_ang = features0[(keypoints) * 4 : 84]
            bone_ang_dist = features0[84:]
        else:
            raise ValueError("INFO: features size is not in range.")

        print("blocks: ", blocks)
        print(f"features size: {len(features)}")

        features = np.array(np.split(np.array(features), blocks))
        for f in features:
            p = [int(f[0] * w), int(f[1] * h)]
            # print(p)
            cv2.circle(img, p, points_tickness, points_COLOR, -1)
            # if add_distance_angle_features_in_two_sequences:
            #     print("dists: ", f[2])
            #     print("angles: ", f[3])

        if len(line_dist_ang) > 1:
            show_line_angle(img, features, line_dist_ang, w, h)
        if len(bone_ang_dist) > 1:
            show_bones_angle(img, features, bone_ang_dist, w, h)

        # cv2.imshow("win", img)
        cv2.imshow("frame", img)
        # cv2.imshow("original image", skeletal_img)
        # cv2.imshow("high quality image1", skeletal_img1)
        # cv2.imshow("high quality image test", skeletal_img2)
        # cv2.imshow("frame_test", frame_test)

        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()


def show_in_each_folders(dir_patients):
    dir_data = dir_patients
    base_folders = [
        name
        for name in os.listdir(dir_patients)
        if os.path.isdir(os.path.join(dir_patients, name))
    ]
    for folder_patient_id in base_folders:
        base_dir = os.path.join(dir_patients, folder_patient_id)
        folders_patient_date = [
            name
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
        ]
        for folder_date in folders_patient_date:
            base_dir = os.path.join(dir_patients, folder_patient_id, folder_date)
            folders_patient_exercise = [
                name
                for name in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, name))
            ]
            for folder_exercise in folders_patient_exercise:
                base_dir = os.path.join(
                    dir_patients, folder_patient_id, folder_date, folder_exercise
                )
                final_folder = [
                    name
                    for name in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, name))
                ][0]
                final_folder_images = [
                    os.path.join(base_dir, final_folder, name)
                    for name in os.listdir(os.path.join(base_dir, final_folder))
                ]
                data_file_name = (
                    folder_patient_id
                    + "_"
                    + folder_date
                    + "_"
                    + folder_exercise
                    + "_data.File"
                )
                with open(
                    os.path.join(
                        dir_data, folder_patient_id, folder_date, data_file_name
                    ),
                    "r",
                ) as data_file:
                    arr = data_file.readlines()
                    show_features_on_selected_images(final_folder_images, arr)
                    print(f"Test on {folder_exercise} is done...")


if __name__ == "__main__":
    keypoints = 17
    points_tickness = 2
    points_COLOR = (50, 50, 250)

    base_dir = "C:/Users/mohsen/Desktop/Postdoc_Upa/Datasets/GaitAnalysis"
    tracking_folder_name = "dst"  # 'tracked_features_100'
    detection_folder_name = "new_final_100"

    show_with_tracking = True
    show_with_70 = False

    if show_with_tracking:
        dir_patients = f"{base_dir}/{tracking_folder_name}"
        if show_with_70:
            show_with_final_file(dir_patients)
        else:
            show_with_34_feats(dir_patients)
    else:
        dir_patients = f"{base_dir}/{detection_folder_name}"
        add_distance_angle_features_in_two_sequences = True
        add_distance_angle_features_in_one_sequence = True
        add_speed_angle_features_in_two_sequences = True

        show_in_each_folders(dir_patients)
