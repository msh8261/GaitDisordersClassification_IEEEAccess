# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103, e1101
# import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
# from sklearn import metrics
# from random import shuffle
from sklearn.utils import shuffle

# import torch.nn as nn
import config.config_train as config
from src.dataset import GaitData
from src.load import LoadData

# from torch_snippets import Report, show


device = torch.device("cuda" if torch.cuda.is_available() else "cup")


def filters(im, mode="sharpen"):
    # remove noise
    im = cv2.GaussianBlur(im, (3, 3), 0)
    if mode == "laplacian":
        # convolute with proper kernels
        im_out = cv2.Laplacian(im, cv2.CV_32F)
    elif mode == "sobelx":
        im_out = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)  # x
    elif mode == "sobely":
        im_out = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=5)  # y
    elif mode == "sharpen":
        # kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        im_out = cv2.filter2D(src=im, ddepth=-1, kernel=kernel)
    return im_out


def convert_to_spect(img):
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    phase_spectrum = np.angle(dft_shift)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift))
    return magnitude_spectrum


def test_visualize_image(test_dataset):
    labels, preds, ims, _ims, specs, sharps = [], [], [], [], [], []

    print("============ Test Results =============")
    test_dataset = shuffle(test_dataset)
    for ix in range(len(test_dataset)):
        im, label = test_dataset[ix]
        # data_input = torch.autograd.Variable(torch.tensor(im[None])).to(device)
        img_str = im * 255
        gray = img_str.astype(np.uint8)
        img_cv = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # print(img_cv)
        print(img_cv.shape)
        cv2.imshow("show", img_cv)
        cv2.waitKey(300)

        labels.extend(label)
        im = im * 255
        ims.append(im)
        spec = convert_to_spect(im)
        specs.append(spec)
        # laplacia, sharpen, sobelx, sobely
        imf = filters(im, mode="sobelx")
        sharps.append(imf)


def test_visualize(test_dataset):
    labels, preds, ims, _ims, specs, sharps = [], [], [], [], [], []

    print("============ Test Results =============")
    # test_dataset = shuffle(test_dataset)
    for ix in range(len(test_dataset)):
        im, label = test_dataset[ix]
        data_input = torch.autograd.Variable(torch.tensor(im[None])).to(device)
        labels.extend(label)
        im = im
        ims.append(im)
        spec = convert_to_spect(im)
        specs.append(spec)
        # laplacia, sharpen, sobelx, sobely
        imf = filters(im, mode="sobelx")
        sharps.append(imf)

    # fig, ax = plt.subplots(3, len(labels), figsize=(12, 8))
    # for i in range(len(labels)):
    #     show(ims[i], ax=ax[0][i], title=f'{labels[i]}')
    #     show(specs[i], ax=ax[1][i], title=f'sp')
    #     show(sharps[i], ax=ax[2][i], title=f"f")

    # fig.tight_layout()
    # plt.show()

    # fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    # ax[0][0].plot(ims[0], label=f'{labels[0]}')
    # ax[0][0].legend()
    # ax[0][1].plot(ims[1], label=f'{labels[1]}')
    # ax[0][1].legend()
    # ax[1][0].plot(ims[2], label=f'{labels[2]}')
    # ax[1][0].legend()
    # ax[1][1].plot(ims[3], label=f'{labels[3]}')
    # ax[1][1].legend()
    # fig.tight_layout()
    # plt.show()

    # fig, ax = plt.subplots(24, figsize=(12, 10))
    # for i in range(24):
    #     ax[i].plot(ims[i], label=f'{labels[i]}')
    #     ax[i].legend(f'{labels[i]}',loc="upper right")

    fig, ax = plt.subplots(3, 6, figsize=(12, 6))
    ax[0][0].plot(ims[12], label=f"{labels[12]}")
    ax[0][0].legend(f"{labels[12]}", loc="upper right")
    ax[0][1].plot(ims[12].T[0], label="nose")
    ax[0][1].legend("nose", loc="upper right")
    ax[0][2].plot(ims[12].T[33], label="left wrist")
    ax[0][2].legend("left wrist", loc="upper right")
    ax[0][3].plot(ims[12].T[37], label="right wrist")
    ax[0][3].legend("right wrist", loc="upper right")
    ax[0][4].plot(ims[12].T[61], label="right ankle")
    ax[0][4].legend("right ankle", loc="upper right")
    ax[0][5].plot(ims[12].T[65], label="left ankle")
    ax[0][5].legend("left ankle", loc="upper right")

    ax[1][0].plot(ims[6], label=f"{labels[6]}")
    ax[1][0].legend(f"{labels[6]}", loc="upper right")
    ax[1][1].plot(ims[6].T[0], label="x-nose")
    ax[1][2].plot(ims[6].T[33], label="x-left-wrist")
    ax[1][3].plot(ims[6].T[37], label="x-right-wrist")
    ax[1][4].plot(ims[6].T[61], label="right ankle")
    ax[1][4].legend("right ankle", loc="upper right")
    ax[1][5].plot(ims[6].T[65], label="left ankle")
    ax[1][5].legend("left ankle", loc="upper right")

    ax[2][0].plot(ims[20], label=f"{labels[20]}")
    ax[2][0].legend(f"{labels[20]}", loc="upper right")
    ax[2][1].plot(ims[20].T[0], label="x-nose")
    ax[2][2].plot(ims[20].T[33], label="x-left-wrist")
    ax[2][3].plot(ims[20].T[37], label="x-right-wrist")
    ax[2][4].plot(ims[20].T[61], label="right ankle")
    ax[2][4].legend("right ankle", loc="upper right")
    ax[2][5].plot(ims[20].T[65], label="left ankle")
    ax[2][5].legend("left ankle", loc="upper right")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    input_size = config.params["input_size"]

    train_dataset_path = config.params["train_dataset_path"]
    X_test_path = os.path.join(train_dataset_path, "Xtest.File")
    y_test_path = os.path.join(train_dataset_path, "ytest.File")

    ld_ts = LoadData(X_test_path, y_test_path, 0)
    X_test = ld_ts.get_X()
    y_test = ld_ts.get_y()
    test_dataset = GaitData(X_test, y_test)

    test_visualize(test_dataset)
