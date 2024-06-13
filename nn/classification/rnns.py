# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn import svm


class RNN(pl.LightningModule):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, learning_rate=0.001
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out


class RNN_new(pl.LightningModule):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, learning_rate=0.001
    ):
        super().__init__()
        self.input_size = input_size - 16
        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size * 3, output_size)

    def forward(self, x):
        x1 = x[:, :100, : self.input_size]
        x2 = x[:, 100:200, : self.input_size]
        x3 = x[:, 200:, : self.input_size]
        out1, _ = self.rnn(x1)
        out1 = out1[:, -1, :]
        out2, _ = self.rnn(x2)
        out2 = out2[:, -1, :]
        out3, _ = self.rnn(x3)
        out3 = out3[:, -1, :]
        out = torch.cat((out1, out2, out3), dim=1)
        out = self.fc(out)

        return out


class RNN_1(pl.LightningModule):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, learning_rate=0.001
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size - 68,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x[:, :, 68:]
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class GRU(pl.LightningModule):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, learning_rate=0.001
    ):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(input_size, input_size),
            # nn.LeakyReLU(),
            nn.ReLU(),
            # nn.Dropout(0.1),
        )
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.1,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fnn(x)
        out, state = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTM(pl.LightningModule):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, learning_rate=0.001
    ):
        super().__init__()
        self.bidirectional = False
        self.fnn = nn.Sequential(
            nn.Linear(input_size, input_size),
            # nn.LeakyReLU(),
            nn.ReLU(),
            # nn.Dropout(0.1),
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.1,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        if self.bidirectional:
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # self.lstm.flatten_parameters()
        # if not hasattr(self, '_flattened'):
        #     self.lstm.flatten_parameters()
        #     setattr(self, '_flattened', True)
        x = self.fnn(x)
        out, state = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTM_new(pl.LightningModule):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, learning_rate=0.001
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.bidirectional = False  # it work better than bidir
        self.input_size = input_size - 16
        ratio = 2

        # self.clf = svm.SVC()

        self.fnn = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            nn.LeakyReLU(),
            # nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.input_size * ratio,
            num_layers=num_layers,
            batch_first=True,  # should be True to increase the accuracy'
            bidirectional=self.bidirectional,
        )

        if self.bidirectional:
            self.fc = nn.Linear(self.input_size * ratio * 2 * 3, output_size)
        else:
            self.fc = nn.Linear(self.input_size * ratio * 3, output_size)

    def forward(self, x):
        # if not hasattr(self, '_flattened'):
        #     self.lstm.flatten_parameters()
        #     setattr(self, '_flattened', True)

        x1 = self.fnn(x[:, :100, : self.input_size])
        x2 = self.fnn(x[:, 100:200, : self.input_size])
        x3 = self.fnn(x[:, 200:, : self.input_size])

        # ## head_points: set input: 20
        # x1 = self.fnn(x[:, :100, :20])
        # x2 = self.fnn(x[:, 100:200, :20])
        # x3 = self.fnn(x[:, 200:, :20])

        # ## hands_points: set input: 24
        # x1 = self.fnn(torch.index_select(x[:, :100, :68],2,torch.tensor([20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]).to("cuda")))
        # x2 = self.fnn(torch.index_select(x[:, 100:200, :68],2,torch.tensor([20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]).to("cuda")))
        # x3 = self.fnn(torch.index_select(x[:, 200:, :68],2,torch.tensor([20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]).to("cuda")))

        # ## legs_points: set input: 24
        # x1 = self.fnn(torch.index_select(x[:, :100, :68],2,torch.tensor([44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]).to("cuda")))
        # x2 = self.fnn(torch.index_select(x[:, 100:200, :68],2,torch.tensor([44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]).to("cuda")))
        # x3 = self.fnn(torch.index_select(x[:, 200:, :68],2,torch.tensor([44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]).to("cuda")))

        out1, state1 = self.lstm(x1)
        out1 = out1[:, -1, :]
        out2, state2 = self.lstm(x2)
        out2 = out2[:, -1, :]
        out3, state3 = self.lstm(x3)
        out3 = out3[:, -1, :]
        out = torch.cat((out1, out2, out3), dim=1)

        # out = self.fnn(out)
        out = self.fc(out)
        return out
