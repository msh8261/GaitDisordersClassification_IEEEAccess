# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103, e1101, e1136
import numpy as np
# import config.config_data as config
import torch
import yaml

import src.kalman as kalman
from prepare_dataset.tools.decorators import (countcall, dataclass, logger,
                                              timeit)

with open("config/config_data.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

num_keypoints = config["dataset_params"]["num_keypoints"]


def point2xyv(kp):
    kp = np.array(kp)
    x = kp[0::3].astype(int)
    y = kp[1::3].astype(int)
    v = kp[2::3].astype(int)  # visibility, 0 = Not visible, 0 != visible
    return x, y, v


def initialize_kalman(num_keypoints, dt=1):
    """initialize kalman filter for 17 keypoints"""
    list_KFs = []
    for i in range(num_keypoints):
        KF = kalman.KF2d(dt=dt)  # time interval: '1 frame'
        init_P = 1 * np.eye(4, dtype=np.float64)  # Error cov matrix
        init_x = np.array(
            [0, 0, 0, 0], dtype=np.float64
        )  # [x loc, x vel, y loc, y vel]
        dict_KF = {"KF": KF, "P": init_P, "x": init_x}
        list_KFs.append(dict_KF)
    return list_KFs, KF


def points_tracking(list_KFs, KF, keypoints):
    list_estimate = []  # kf filtered keypoints
    if len(keypoints) > 0:
        keypoints = keypoints[0].detach().cpu().numpy()
        for i in range(len(keypoints)):
            # print(keypoints[i])
            kx = keypoints[i][0]
            ky = keypoints[i][1]
            z = np.array([kx, ky], dtype=np.float64)

            KF = list_KFs[i]["KF"]
            x = list_KFs[i]["x"]
            P = list_KFs[i]["P"]

            x, P, filtered_point = KF.process(x, P, z)

            list_KFs[i]["KF"] = KF
            list_KFs[i]["x"] = x
            list_KFs[i]["P"] = P

            # visibility
            v = 0 if filtered_point[0] == 0 and filtered_point[1] == 0 else 2
            list_estimate.extend(list(filtered_point) + [v])  # x,y,v

    return list_estimate


@logger
class SkeletonTracker(object):
    def __init__(self, device) -> None:
        self.device = device

    def get_tracked_missing_points(self, persons):
        dt = 1
        list_KFs, KF = initialize_kalman(num_keypoints, dt)
        list_estimate = points_tracking(list_KFs, KF, persons)
        list_estimate = [
            [
                list_estimate[i * 3],
                list_estimate[(i * 3) + 1],
                list_estimate[(i * 3) + 2],
            ]
            for i in range(len(list_estimate))
            if i < len(list_estimate) / 3
        ]
        tracked_points = torch.tensor([list_estimate]).to(self.device)
        return tracked_points

    def get_tracked_one_frame_missing_detection(self, persons, last_keypoint_kalman):
        dt = 1
        if len(persons) > 0:
            persons = torch.unsqueeze(persons[0], dim=0)
        else:
            persons = last_keypoint_kalman
        list_KFs, KF = initialize_kalman(num_keypoints, dt)
        list_estimate = points_tracking(list_KFs, KF, persons)
        list_estimate = [
            [
                list_estimate[i * 3],
                list_estimate[(i * 3) + 1],
                list_estimate[(i * 3) + 2],
            ]
            for i in range(len(list_estimate))
            if i < len(list_estimate) / 3
        ]
        tracked_points = torch.tensor([list_estimate]).to(self.device)
        return tracked_points

    def get_tracked_five_frames(self, persons, last_keypoint_kalman, cnt):
        dt = 5
        if len(persons) > 0 and cnt % dt == 0:
            persons = torch.unsqueeze(persons[0], dim=0)
        else:
            persons = last_keypoint_kalman
        list_KFs, KF = initialize_kalman(num_keypoints, dt)
        list_estimate = points_tracking(list_KFs, KF, persons)
        list_estimate = [
            [
                list_estimate[i * 3],
                list_estimate[(i * 3) + 1],
                list_estimate[(i * 3) + 2],
            ]
            for i in range(len(list_estimate))
            if i < len(list_estimate) / 3
        ]
        tracked_points = torch.tensor([list_estimate]).to(self.device)
        return tracked_points
