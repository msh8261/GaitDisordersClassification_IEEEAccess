# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103
import pytorch_lightning as pl
import torch
import torch.nn as nn


class MLP(pl.LightningModule):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, learning_rate=0.001
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        # x = x.squeeze(dim=2)
        out = self.mlp(x)[:, -1]
        return out


class MLP_new(pl.LightningModule):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, learning_rate=0.001
    ):
        super().__init__()
        self.input_size = input_size - 16
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 120),
        )
        self.fc = nn.Linear(120 * 3, output_size)

    def forward(self, x):
        x1 = x[:, :100, : self.input_size]
        x2 = x[:, 100:200, : self.input_size]
        x3 = x[:, 200:, : self.input_size]
        # x = x.squeeze(dim=2)
        out1 = self.mlp(x1)[:, -1]
        out2 = self.mlp(x2)[:, -1]
        out3 = self.mlp(x3)[:, -1]
        out = torch.cat((out1, out2, out3), dim=1)
        out = self.fc(out)
        return out
