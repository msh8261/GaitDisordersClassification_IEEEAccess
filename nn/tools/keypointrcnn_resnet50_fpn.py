# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103
import torch
import torchvision

import config.config_data as config


class keypointrcnn_resnet50_fpn:
    @staticmethod
    def create():
        # create a model object from the keypointrcnn_resnet50_fpn class
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            pretrained=True, num_keypoints=config.params["num_keypoints"], min_size=550
        )  # 800 is default
        # call the eval() method to prepare the model for inference mode.
        # set the computation device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load the modle on to the computation device and set to eval mode
        model.to(device).eval()

        return model, device
