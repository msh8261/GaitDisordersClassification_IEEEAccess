# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103

import math
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

import config.config_train as config

activations = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "selu": nn.SELU(),
    "elu": nn.ELU(),
    "LeakyReLU": nn.LeakyReLU(0.1),
    "rrelu": nn.RReLU(0.1, 0.3),
}


class MLP(nn.Module):
    """Multilayer perceptron."""

    def __init__(self, in_features, hidden_features, out_features, p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)  # (n_samples, n_patches , hidden_features)
        x = self.act(x)  # (n_samples, n_patches , hidden_features)
        x = self.drop(x)  # (n_samples, n_patches , hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches , out_features)
        x = self.drop(x)  # (n_samples, n_patches , out_features)

        return x


class AutoEncoderMLP(pl.LightningModule):
    def __init__(self, W_img, H_img):
        super(AutoEncoderMLP, self).__init__()
        # self.save_hyperparameters()
        self.H_img = H_img
        self.W_img = W_img
        self.dropout = nn.Dropout2d(config.params["dropout"])
        self.activation_function = activations[config.params["acf_indx"]]
        self.bottleneck = config.params["bottleneck"]
        mlp_ratio = 4
        hidden_size = int(self.W_img * mlp_ratio)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.glu = nn.GELU()

        self.encoder = nn.Sequential(
            nn.Linear(self.W_img * self.H_img, hidden_size),
            self.activation_function,
            self.dropout,
            nn.Linear(hidden_size, 512),
            self.activation_function,
            nn.Linear(512, self.bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.bottleneck, 512),
            self.activation_function,
            nn.Linear(512, hidden_size),
            self.activation_function,
            self.dropout,
            nn.Linear(hidden_size, self.input_size * self.sequence_length),
        )

        self.norm = nn.LayerNorm(self.input_size, eps=1e-6)

    def forward(self, x):
        print("@@@@@@@@@@@@@@@@@@@@@@@@")
        print(x.shape)
        print("@@@@@@@@@@@@@@@@@@@@@@@@")
        x = x.reshape(len(x), -1)
        encoded = self.encoder(x)
        # decoded = self.tanh(self.decoder(encoded))
        decoded = self.sigmoid(self.decoder(encoded))
        decoded = decoded.reshape(len(decoded), self.sequence_length, self.input_size)
        extracted_features = encoded
        return extracted_features
