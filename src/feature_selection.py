# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103
import math
import os
import random
from math import atan2, degrees

# from stat import FILE_ATTRIBUTE_REPARSE_POINT
import numpy as np
from scipy.spatial import distance
from sklearn import preprocessing
from torchvision import transforms as T

min_dist_from_frame = 10


def get_keypoint(keypoints):
    Params = {}
    Params["left_eye"] = keypoints[1]
    Params["right_eye"] = keypoints[2]

    Params["left_ear"] = keypoints[3]
    Params["right_ear"] = keypoints[4]

    Params["left_shoulder"] = keypoints[5]
    Params["right_shoulder"] = keypoints[6]

    Params["left_elbow"] = keypoints[7]
    Params["right_elbow"] = keypoints[8]

    Params["left_wrist"] = keypoints[9]
    Params["right_wrist"] = keypoints[10]

    Params["left_hip"] = keypoints[11]
    Params["right_hip"] = keypoints[12]

    Params["left_knee"] = keypoints[13]
    Params["right_knee"] = keypoints[14]

    Params["left_ankle"] = keypoints[15]
    Params["right_ankle"] = keypoints[16]
    return Params


def image_scaling(frame, image_need_crop, scale_w, scale_h):
    # make a copy of the frame
    img = frame.copy()
    h, w = img.shape[:2]
    if image_need_crop:
        w = int(w * scale_w)
        h = int(h * scale_h)
        img = img[0:h, 0:w]
        # img = img[0:h, w:w*2]
    return h, w, img


def input_for_model(img, device):
    # preprocess the input image
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)
    # add one dim with unsqueeze
    return img_tensor.unsqueeze_(0)


def filter_persons(model_output, person_thresh):
    persons = {}
    p_indicies = [i for i, s in enumerate(model_output["scores"]) if s > person_thresh]
    for i in p_indicies:
        desired_kp = model_output["keypoints"][i][:].to("cpu")
        persons[i] = desired_kp
    return (persons, p_indicies)


def check_to_get_all_features_available_in_image(h, w, keypoints):
    if len(keypoints) == 0:
        return 0, [0]
    arr = keypoints[0].detach().cpu().numpy()
    arr = [[a for a in b] for b in arr]
    res = [
        [
            (
                1
                if (
                    min_dist_from_frame < arr[i][0] < (w - min_dist_from_frame)
                    and min_dist_from_frame < arr[i][1] < (h - min_dist_from_frame)
                )
                else 0
            )
        ]
        for i in range(len(arr))
    ]
    res = [item for sublist in res for item in sublist]
    flag = all(res)
    return flag, arr


def normalize_values_from_image(arr, h, w):
    arr = [[float(arr[i][0] / w), float(arr[i][1] / h)] for i in range(len(arr))]
    return arr


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def normalizeAngle(angle_degree):
    newAngle = angle_degree
    if newAngle <= -180:
        newAngle += 360
    if newAngle > 180:
        newAngle -= 360
    return newAngle


def Angle_Btw_2Vectors(vec1, vec2):
    ## angle of two vectors: cos-1 [ (vec1 · vec2) / (|vec1| |vec1|) ]
    return degrees(angle(vec1, vec2))


def Angle_Btw_2Points(pointA, pointB):
    changeInX = pointB[0] - pointA[0]
    changeInY = pointB[1] - pointA[1]
    # normalize between 0 and 1
    return (degrees(atan2(changeInY, changeInX)) + 180) / 360


def Angle_Btw_3Points(pointA, point_Center, pointB):
    vector1 = [(pointA[0] - point_Center[0]), (pointA[1] - point_Center[1])]
    vector2 = [(pointB[0] - point_Center[0]), (pointB[1] - point_Center[1])]
    length1 = math.sqrt(vector1[0] * vector1[0] + vector1[1] * vector1[1])
    length2 = math.sqrt(vector2[0] * vector2[0] + vector2[1] * vector2[1])
    return (
        round(
            normalizeAngle(
                degrees(
                    math.acos(
                        (vector1[0] * vector2[0] + vector1[1] * vector2[1])
                        / (length1 * length2 + 1e-6)
                    )
                )
            ),
            2,
        )
        / 180.0
    )


# def length_2Points(pointA, pointB):
#   changeInX = pointB[0] - pointA[0]
#   changeInY = pointB[1] - pointA[1]
#   return np.sqrt(pow(changeInX, 2) + pow(changeInY, 2))


# def Angle_Btw_3Points_old(pointA, pointB, pointC):
#     #arccos((P12^2 + P13^2 - P23^2) / (2 * P12 * P13))
#     #sqrt((P1x - P2x)^2 + (P1y - P2y)^2)
#     P12 = length_2Points(pointA, pointB)
#     P13 = length_2Points(pointA, pointC)
#     P23 = length_2Points(pointB, pointC)
#     return (np.arccos(pow(P12, 2) + pow(P13, 2) - pow(P23, 2)) / (2 * P12 * P13))


# def Angle_Btw_2Points(pointA, pointB):
#   changeInX = pointB[0] - pointA[0]
#   changeInY = pointB[1] - pointA[1]
#   radians_ = atan2(changeInY,changeInX)
#   angle = degrees(radians_)
#   if (radians_ < 0):
#     angle += 360
#   # normalize between 0 and 1
#   return angle//360


def add_distance_angle_of_keypoints_in_two_sequences(arr_last, arr_current):
    arr_dist = [
        distance.euclidean(arr_last[i], arr_current[i]) for i in range(len(arr_last))
    ]
    arr_angle = [
        Angle_Btw_2Points(arr_last[i], arr_current[i]) for i in range(len(arr_last))
    ]
    final_arr = [
        [arr_current[i][0], arr_current[i][1], arr_dist[i], arr_angle[i]]
        for i in range(len(arr_current))
    ]
    final_arr = [item for sublist in final_arr for item in sublist]
    return final_arr


def add_speed_angle_of_keypoints_in_two_sequences(dtime, arr_last, arr_current):
    if dtime <= 0:
        dtime = 1
    arr_speed = [
        (distance.euclidean(arr_last[i], arr_current[i]) / dtime)
        for i in range(len(arr_last))
    ]
    arr_speed = preprocessing.normalize([arr_speed], norm="l2")[0]
    arr_angle = [
        Angle_Btw_2Points(arr_last[i], arr_current[i]) for i in range(len(arr_last))
    ]
    final_arr = [
        [arr_current[i][0], arr_current[i][1], arr_speed[i], arr_angle[i]]
        for i in range(len(arr_current))
    ]
    # final_arr = [[arr_speed[i], arr_angle[i]] for i in range(len(arr_current))]
    final_arr = [item for sublist in final_arr for item in sublist]
    return final_arr


def add_distance_angle_of_symetric_keypoints_in_a_sequence(arr, keypoints):
    left_eye = keypoints[1]
    right_eye = keypoints[2]

    left_ear = keypoints[3]
    right_ear = keypoints[4]

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    left_elbow = keypoints[7]
    right_elbow = keypoints[8]

    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    left_hip = keypoints[11]
    right_hip = keypoints[12]

    left_knee = keypoints[13]
    right_knee = keypoints[14]

    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    left_side = [
        left_eye,
        left_ear,
        left_shoulder,
        left_elbow,
        left_wrist,
        left_hip,
        left_knee,
        left_ankle,
    ]
    right_side = [
        right_eye,
        right_ear,
        right_shoulder,
        right_elbow,
        right_wrist,
        right_hip,
        right_knee,
        right_ankle,
    ]

    features = [[left_side[i], right_side[i]] for i in range(len(right_side))]
    arr_dist = [
        distance.euclidean(left_side[i], right_side[i]) for i in range(len(left_side))
    ]
    arr_angle = [
        Angle_Btw_2Points(left_side[i], right_side[i]) for i in range(len(left_side))
    ]
    features = [[arr_dist[i], arr_angle[i]] for i in range(len(left_side))]

    features = [item for sublist in features for item in sublist]
    arr.extend(features)
    return arr


# displacement of pairwise joints
def add_displacement_pairwise_joints(arr, keypoints):
    # points = get_keypoint(keypoints)
    # nose = points["nose"]
    # left_shoulder = points["left_shoulder"]
    # right_shoulder = points["right_shoulder"]
    # left_hip = points["left_hip"]
    # right_hip = points["right_hip"]
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]

    nose_left_elbow_dist = distance.euclidean(nose, keypoints[7])
    nose_right_elbow_dist = distance.euclidean(nose, keypoints[8])

    nose_left_wrist_dist = distance.euclidean(nose, keypoints[9])
    nose_right_wrist_dist = distance.euclidean(nose, keypoints[10])

    nose_left_knee_dist = distance.euclidean(nose, keypoints[13])
    nose_right_knee_dist = distance.euclidean(nose, keypoints[14])

    nose_left_ankle_dist = distance.euclidean(nose, keypoints[15])
    nose_right_ankle_dist = distance.euclidean(nose, keypoints[16])

    left_shoulder_left_wrist_dist = distance.euclidean(left_shoulder, keypoints[9])
    right_shoulder_right_wrist_dist = distance.euclidean(right_shoulder, keypoints[10])

    left_hip_left_ankle_dist = distance.euclidean(left_hip, keypoints[15])
    right_hip_right_ankle_dist = distance.euclidean(right_hip, keypoints[16])

    features = [
        nose_left_elbow_dist,
        nose_right_elbow_dist,
        nose_left_wrist_dist,
        nose_right_wrist_dist,
        nose_left_knee_dist,
        nose_right_knee_dist,
        nose_left_ankle_dist,
        nose_right_ankle_dist,
        left_shoulder_left_wrist_dist,
        right_shoulder_right_wrist_dist,
        left_hip_left_ankle_dist,
        right_hip_right_ankle_dist,
    ]

    arr.extend(features)
    return arr


#  angles of the selected body bones
def angles_selected_body_bones(arr, keypoints):

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    left_elbow = keypoints[7]
    right_elbow = keypoints[8]

    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    left_hip = keypoints[11]
    right_hip = keypoints[12]

    left_knee = keypoints[13]
    right_knee = keypoints[14]

    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    ang1 = normalizeAngle(
        (Angle_Btw_3Points(right_shoulder, left_shoulder, left_elbow))
    )
    ang2 = normalizeAngle(
        (Angle_Btw_3Points(left_shoulder, right_shoulder, right_elbow))
    )

    ang3 = normalizeAngle((Angle_Btw_3Points(right_shoulder, right_elbow, right_wrist)))
    ang4 = normalizeAngle((Angle_Btw_3Points(left_shoulder, left_elbow, left_wrist)))

    ang5 = normalizeAngle((Angle_Btw_3Points(right_hip, left_hip, left_knee)))
    ang6 = normalizeAngle((Angle_Btw_3Points(left_hip, right_hip, right_knee)))

    ang7 = normalizeAngle((Angle_Btw_3Points(right_hip, right_knee, right_ankle)))
    ang8 = normalizeAngle((Angle_Btw_3Points(left_hip, left_knee, left_ankle)))

    # ang1 = (Angle_Btw_3Points(right_shoulder, left_shoulder, left_elbow))
    # ang2 = (Angle_Btw_3Points(left_shoulder, right_shoulder, right_elbow))

    # ang3 = (Angle_Btw_3Points(right_shoulder, right_elbow, right_wrist))
    # ang4 = (Angle_Btw_3Points(left_shoulder, left_elbow, left_wrist))

    # ang5 = (Angle_Btw_3Points(right_hip, left_hip, left_knee))
    # ang6 = (Angle_Btw_3Points(left_hip, right_hip, right_knee))

    # ang7 = (Angle_Btw_3Points(right_hip, right_knee, right_ankle))
    # ang8 = (Angle_Btw_3Points(left_hip, left_knee, left_ankle))

    features = [ang1, ang2, ang3, ang4, ang5, ang6, ang7, ang8]

    arr.extend(features)
    return arr


# if __name__ == "__main__":
#     for i in range(50):
#         # vec1 = [random.randint(200, 400), random.randint(20, 60)]
#         # vec2 = [random.randint(20, 60), random.randint(200, 400)]
#         p1 = [1, 2]
#         p2 = [1, 0]
#         print(Angle_Btw_2Vectors(p1, p2))

#         # 90 degree
#         print("90: ", normalizeAngle((Angle_Btw_3Points([50,0], [50, 50], [100, 50]))))
#         # 0 degree
# print("0: ", normalizeAngle((Angle_Btw_3Points([100,50], [50, 50], [100, 50]))))
# # 45 degree
# print("45: ", normalizeAngle((Angle_Btw_3Points([0,0], [0,100], [100,0]))))
# # 135 degree
# print("135: ", normalizeAngle((Angle_Btw_3Points([200,0], [200,100], [100,200]))))
# print("====")

#     vec = [0.2676286103922626,0.25,0.21880051797725197,0.21696072994705193,0.2141028883138093,0.25,0.21880051797725247,0.21696072994705004,0.2676286103922626,0.25,0.2714013503711642,0.22343181148366148,0.2323241836647824,0.18654618225525696,0.10487841749002064,0.914756125171978,0.2998793958338487,0.1755090903470516,0.48173149870607196,0.75,0.39835627548985303,0.19485192767591464,0.2097568349800399,0.08524387482802022,0.10487841749001996,0.08524387482802022,0.10487841749002222,0.41475612517198185,0.11616209183239166,0.1865461822552602,0.11616209183239166,0.3134538177447398,0.05352572207845199,0.25,0.011605834824119369,0.9689168412349267,0.030692872498577887,0.9765576522635344,0.06736163995333103,0.9732675659490923,0.1699339100578677,0.9702963851828882,0.27100783791083677,0.9585291763645816,0.03668081258247499,0.9705146268546817,0.026565464895635715,1.0,0.008826136027023963,0.914756125171978,0.8963333333333333,0.8651666666666666,0.9899444444444444,0.8779444444444444,0.4301111111111111,0.5415,0.9945555555555556,0.9778333333333333,0.10262987422809537,0.10943032435375334,0.1536648627703409,0.15253690574774642,0.28036658625549016,0.27928572549814407,0.38986597666350487,0.39418525093356693,0.11583714102705495,0.09629826626564911,0.2212421230706759,0.21887188649124975
# ]
#     print(len(vec))
#     print(max(vec))
#     print(min(vec))
