# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103
import pytorch_lightning as pl
import torch
import torch.nn as nn

import config.config_data as config

WINDOW_SIZE = config.params["WINDOW_SIZE"]
num_exercise = 3


class CNN(pl.LightningModule):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, learning_rate=0.001
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        sequences = WINDOW_SIZE * num_exercise
        in_c = sequences

        self.conv_layer1 = CNN._conv_layer_set(in_c, input_size, 3)
        self.conv_layer2 = CNN._conv_layer_set(input_size, 64, 3)
        self.conv_layer3 = CNN._conv_layer_set(64, 128, 5)
        self.conv_layer4 = CNN._conv_layer_set(128, 256, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 1, 128)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, output_size)
        # self.softmax = nn.Softmax()
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def _conv_layer_set(in_c, out_c, kernel_size):
        conv_layer = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
        )
        return conv_layer

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.softmax(self.fc2(out))

        return out


class CNN2D(pl.LightningModule):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, learning_rate=0.001
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        sequences = WINDOW_SIZE * num_exercise
        in_c = 1
        out_c1 = 30
        out_c2 = 30

        self.conv_layer1 = CNN2D._conv_layer_set(in_c, out_c1, (4, 4))
        self.conv_layer2 = CNN2D._conv_layer_set(in_c, out_c2, (300, 1))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(300 * 84 * 30, 128)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, output_size)
        # self.softmax = nn.Softmax()
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def _conv_layer_set(in_c, out_c, kernel_size):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
        return conv_layer

    def forward(self, x):
        print("==================")
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        print(x.shape)
        out1 = self.conv_layer1(x)
        out2 = self.conv_layer2(x)
        out = torch.cat((out1, out2), dim=1)

        print(out.shape)
        print("==================")
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.softmax(self.fc2(out))

        return out
