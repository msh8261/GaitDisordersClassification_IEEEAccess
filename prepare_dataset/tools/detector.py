# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103, e1101, e1136
import cv2
import torch
import yaml

import src.feature_selection as fs
from nn.tools.fasterrcnn_resnet50_fpn import FasterRCNN_resnet50_fpn
from nn.tools.keypointrcnn_resnet50_fpn import keypointrcnn_resnet50_fpn
from prepare_dataset.tools.decorators import (countcall, dataclass, logger,
                                              timeit)

# import config.config_data as config

with open("config/config_data.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

image_need_crop = config["dataset_params"]["image_need_crop"]
scale_w = config["dataset_params"]["scale_w"]
scale_h = config["dataset_params"]["scale_h"]
person_thresh = config["dataset_params"]["person_thresh"]
num_keypoints = config["dataset_params"]["num_keypoints"]


@logger
class SkeletonDetector(object):
    def __init__(self) -> None:
        self.model, self.device = keypointrcnn_resnet50_fpn.create()

    def get_detector_model(self):
        return self.model

    def get_device(self):
        return self.device

    def get_detected_skeleton_model(self, image):
        tensor_image = fs.input_for_model(image, self.device)
        return self.model(tensor_image)[0]

    def select_best_detection_filter_result(
        self, filtered_image_mix, filtered_image_hist
    ):
        model = self.model
        device = self.device

        h, w, img_mix = fs.image_scaling(
            filtered_image_mix, image_need_crop, scale_w, scale_h
        )
        h, w, img_hist = fs.image_scaling(
            filtered_image_hist, image_need_crop, scale_w, scale_h
        )

        img_tensor_mix = fs.input_for_model(img_mix, device)
        img_tensor_hist = fs.input_for_model(img_hist, device)

        output_mix = model(img_tensor_mix)[0]
        output_hist = model(img_tensor_hist)[0]
        persons_mix, p_inds_mix = fs.filter_persons(output_mix, person_thresh)
        persons_hist, p_inds_hist = fs.filter_persons(output_hist, person_thresh)

        keypoints_scores_mix = output_mix["keypoints_scores"].cpu().detach().numpy()
        keypoints_scores_hist = output_hist["keypoints_scores"].cpu().detach().numpy()
        if (len(persons_hist) == 1) and len(keypoints_scores_hist) > 0:
            if len(keypoints_scores_hist[p_inds_hist][0]) == num_keypoints:
                return persons_hist, p_inds_hist, keypoints_scores_hist, img_hist
        elif (len(persons_mix) == 1) and len(keypoints_scores_mix) > 0:
            return persons_mix, p_inds_mix, keypoints_scores_mix, img_mix
        else:
            return persons_hist, p_inds_hist, keypoints_scores_hist, img_hist

    @staticmethod
    def get_image_with_keypoints(img, persons):
        # observe keypoints on images
        if len(persons) == 1:
            for i in range(len(persons[0])):
                x_p = int(persons[0][i][0].detach().cpu().numpy())
                y_p = int(persons[0][i][1].detach().cpu().numpy())
                cv2.circle(img, (x_p, y_p), 3, (200, 100, 0), -1, 1)
        return img


@logger
class PersonDetector(object):
    def __init__(self) -> None:
        self.model, self.device = FasterRCNN_resnet50_fpn.create()

    def get_detector_model(self):
        return self.model

    def get_device(self):
        return self.device

    def get_detected_skeleton_model(self, image):
        tensor_image = fs.input_for_model(image, self.device)
        return self.model(tensor_image)[0]

    def select_best_detection_filter_result(
        self, filtered_image_mix, filtered_image_hist
    ):
        model = self.model
        device = self.device

        h, w, img_mix = fs.image_scaling(
            filtered_image_mix, image_need_crop, scale_w, scale_h
        )
        h, w, img_hist = fs.image_scaling(
            filtered_image_hist, image_need_crop, scale_w, scale_h
        )

        img_tensor_mix = fs.input_for_model(img_mix, device)
        img_tensor_hist = fs.input_for_model(img_hist, device)

        output_mix = model(img_tensor_mix)[0]
        output_hist = model(img_tensor_hist)[0]
        persons_mix, p_inds_mix = self.filter_persons(output_mix, person_thresh)
        persons_hist, p_inds_hist = self.filter_persons(output_hist, person_thresh)

        scores_mix = output_mix["scores"].cpu().detach().numpy()
        scores_hist = output_hist["scores"].cpu().detach().numpy()
        if (len(persons_hist) == 1) and len(scores_hist) > 0:
            if len(scores_hist[p_inds_hist][0]) == num_keypoints:
                return persons_hist, p_inds_hist, scores_hist, img_hist
        elif (len(persons_mix) == 1) and len(scores_mix) > 0:
            return persons_mix, p_inds_mix, scores_mix, img_mix
        else:
            return persons_hist, p_inds_hist, scores_hist, img_hist

    @staticmethod
    def filter_persons(model_output, person_thresh):
        persons = {}
        p_indicies = [
            i for i, s in enumerate(model_output["scores"]) if s > person_thresh
        ]
        for i in p_indicies:
            desired_kp = model_output["boxes"][i][:].to("cpu")
            persons[i] = desired_kp
        return (persons, p_indicies)

    @staticmethod
    def get_image_with_BBox(img, persons):
        bbox = int(persons[0]["boxes"].detach().cpu().numpy())
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]
        cv2.rectangle(img, (x, y, w, h), (200, 100, 0), 2)
        return img

    @staticmethod
    def crop_image_size_BBox(img, persons):
        bbox = int(persons[0]["boxes"].detach().cpu().numpy())
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]
        croped_img = img[y : y + h, x : x + w]
        return croped_img
