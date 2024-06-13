# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103

import math
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

import config.config_train as config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

activations = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "selu": nn.SELU(),
    "elu": nn.ELU(),
    "LeakyReLU": nn.LeakyReLU(0.1),
    "rrelu": nn.RReLU(0.1, 0.3),
}


dropout = nn.Dropout(config.params["dropout"])
activation_function = activations[config.params["acf_indx"]]
bottleneck = config.params["bottleneck"]
last_layer = config.params["last_layer"]
num_class = config.params["num_class"]


# #Encoder is 2 separate layers of the LSTM RNN
# class Encoder(nn.Module):

#   def __init__(self, embedding_dim):
#     super(Encoder, self).__init__()

#     embedding_dim = embedding_dim

#     self.seq_len, self.n_features = config.params['sequences'], config.params['input_size']
#     self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

#     self.rnn1 = nn.LSTM(
#       input_size=config.params['input_size'],
#       hidden_size=self.hidden_dim,
#       num_layers=1,
#       batch_first=True
#     )
#   # Initializing the hidden numbers of layers
#     self.rnn2 = nn.LSTM(
#       input_size=self.hidden_dim,
#       hidden_size=embedding_dim,
#       num_layers=1,
#       batch_first=True
#     )

#   def forward(self, x):
#     #x = x.reshape((1, self.seq_len, self.n_features))
#     # (1, 300, 104)
#     x, (h_, c_) = self.rnn1(x)
#     # (1, 300, 256)
#     x, (hidden_n, c_) = self.rnn2(x)
#     # x -> (1, 300, 128), hidden_n -> (1, batch, 128)
#     #return hidden_n.reshape((self.n_features, self.embedding_dim))
#     return hidden_n.reshape((hidden_n.shape[1], self.embedding_dim))


# class Decoder(nn.Module):

#   def __init__(self, embedding_dim):
#     super(Decoder, self).__init__()

#     self.seq_len, self.input_dim = config.params['sequences'], embedding_dim
#     self.hidden_dim, self.n_features = 2 * embedding_dim, config.params['input_size']

#     self.rnn1 = nn.LSTM(
#       input_size=embedding_dim,
#       hidden_size=embedding_dim,
#       num_layers=1,
#       batch_first=True
#     )
# #Using a dense layer as an output layer
#     self.rnn2 = nn.LSTM(
#       input_size=embedding_dim,
#       hidden_size=self.hidden_dim,
#       num_layers=1,
#       batch_first=True
#     )

#     self.output_layer = nn.Linear(self.hidden_dim*self.n_features, self.n_features)

#   def forward(self, x):
#     # (batch, input_dim)
#     n_batch = x.shape[0]

#     # repeat -->> (batch*seq_len, input_dim*n_features)
#     x = x.repeat(self.seq_len, self.n_features)

#     x = x.reshape(n_batch, self.n_features, self.seq_len, self.input_dim)

#     x, (hidden_n, cell_n) = self.rnn1(x)
#     # (104 , 300, 128)
#     x, (hidden_n, cell_n) = self.rnn2(x)
#     # (104 , 300, 256)
#     x = x.reshape((self.seq_len, self.hidden_dim*self.n_features))

#     return self.output_layer(x)


# class RecurrentAutoencoder(nn.Module):

#   def __init__(self, embedding_dim=64):
#     super(RecurrentAutoencoder, self).__init__()


#     self.encoder = Encoder(embedding_dim).to(device)
#     self.decoder = Decoder(embedding_dim).to(device)

#     self.classifier = nn.Sequential(
#                                 nn.Linear(embedding_dim, last_layer),
#                                 activation_function,
#                                 dropout,
#                                 nn.Linear(last_layer, num_class),
#                                 )

#   def forward(self, x):
#     encoded = self.encoder(x)
#     decoded = self.decoder(encoded)
#     cls = self.classifier(encoded)
#     return decoded, cls


class GRU(pl.LightningModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        num_layers = 1
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        if not hasattr(self, "_flattened"):
            self.gru.flatten_parameters()
            setattr(self, "_flattened", True)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return out


class LSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super().__init__()
        num_layers = 1
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        if not hasattr(self, "_flattened"):
            self.lstm.flatten_parameters()
            setattr(self, "_flattened", True)
        # self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return out


class RNNsEncoder(pl.LightningModule):
    def __init__(self, model):
        super(RNNsEncoder, self).__init__()
        self.input_size = config.params["input_size"]
        ratio = 2

        self.fnn = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            activation_function,
            dropout,
        )

        if model == "lstm":
            self.model = LSTM(self.input_size, self.input_size * ratio, False)
        elif model == "gru":
            self.model = GRU(self.input_size, self.input_size * ratio)

        self.fc1 = nn.Linear(self.input_size * ratio * 3, bottleneck)
        self.fc2 = nn.Linear(bottleneck, num_class)

    def forward(self, x):

        x1 = self.fnn(x[:, :100, : self.input_size])
        x2 = self.fnn(x[:, 100:200, : self.input_size])
        x3 = self.fnn(x[:, 200:, : self.input_size])

        out1 = self.model(x1)
        out2 = self.model(x2)
        out3 = self.model(x3)
        out = torch.cat((out1, out2, out3), dim=1)

        out = self.fc1(out)
        cls = self.fc2(out)
        return out, cls


class RNNsDecoder(pl.LightningModule):
    def __init__(self, model):
        super(RNNsDecoder, self).__init__()
        self.bidirectional = False  # it work better than bidir
        self.input_size = config.params["input_size"]
        self.sequence_length = config.params["sequences"]
        self.bottleneck = bottleneck
        ratio = 2

        if model == "lstm":
            # self.model = LSTM(self.input_size*ratio*3, self.input_size, False)
            self.model = LSTM(bottleneck, self.input_size, False)
        elif model == "gru":
            # self.model = GRU(self.input_size*ratio*3, self.input_size)
            self.model = GRU(bottleneck, self.input_size)

        self.fnn = nn.Sequential(
            nn.Linear(self.input_size, self.input_size * self.sequence_length),
            nn.Sigmoid(),
        )

    def forward(self, x):
        n_batch = x.shape[0]
        x = x.repeat(self.sequence_length, 1)
        # x = x.reshape(n_batch,  self.sequence_length, self.input_size*2*3)
        x = x.reshape(n_batch, self.sequence_length, self.bottleneck)
        out = self.model(x)
        out = self.fnn(out)
        out = out.reshape(n_batch, self.sequence_length, self.input_size)
        return out


class AutoEncoderRNNs(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        # self.save_hyperparameters()

        self.encoder = RNNsEncoder(model)
        self.decoder = RNNsDecoder(model)

    def forward(self, x):
        encoded, cls = self.encoder(x)
        decoded = self.decoder(encoded)
        return (decoded, cls)
