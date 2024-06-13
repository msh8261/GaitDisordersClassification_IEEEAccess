# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103, e1101, e1136
import os

import cv2

import src.image_filters as fl
from prepare_dataset.tools.decorators import (countcall, dataclass, logger,
                                              timeit)


@logger
class ImageReader(object):
    def __init__(self, image_path) -> None:
        self.image = cv2.imread(image_path)
        self.height, self.width = self.image.shape[:2]

    def get_image(self):
        return self.image

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    @staticmethod
    def write_image(image, save_img_path, filename):
        cv2.imwrite(os.path.join(save_img_path, filename), image)

    @staticmethod
    def get_filtered_image(image, filter_name):
        if filter_name == "hist_colormap":
            filtered_image = fl.apply_hist_colormap_filter(image)
        elif filter_name == "equalhist":
            filtered_image = fl.apply_equalhist_filter(image)
        return filtered_image
