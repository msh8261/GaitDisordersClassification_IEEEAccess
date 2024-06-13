# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

import config.config_train as config

# from torchmetrics.functional import f1_score


TOT_CLASSES = config.params["num_class"]


# lstm classifier definition
class GaitClassificationLSTM(pl.LightningModule):
    # initialise method
    def __init__(
        self, input_features, hidden_dim, num_layers, dropout, learning_rate=0.001
    ):
        super().__init__()
        # save hyperparameters
        self.save_hyperparameters()
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        # self.Linear = nn.Linear(hidden_dim, config.TOT_ACTION_CLASSES)
        # The linear layer that maps from hidden state space to classes
        self.fc = nn.Linear(hidden_dim, TOT_CLASSES)
        # self.bn = nn.BatchNorm1d(32)

    def forward(self, x):
        # x = self.bn(x)
        # invoke lstm layer
        lstm_out, (ht, ct) = self.lstm(x)
        # invoke linear layer
        # out = self.Linear(ht[-1])
        out = self.fc(ht[-1])
        # out = self.fc(lstm_out[:,-1,:])
        return out

    def training_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y = batch
        # reduce dimension
        y = torch.squeeze(y)
        # convert to long
        y = y.long()
        # get prediction
        y_pred = self(x)
        # calculate loss
        loss = F.cross_entropy(y_pred, y)
        # get probability score using softmax
        prob = F.softmax(y_pred, dim=1)
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        # calculate accuracy
        acc = torchmetrics.functional.accuracy(pred, y)
        dic = {"batch_train_loss": loss, "batch_train_acc": acc}
        # log the metrics for pytorch lightning progress bar or any other operations
        self.log("batch_train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("batch_train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        # return loss and dict
        return {"loss": loss, "result": dic}

    def training_epoch_end(self, training_step_outputs):
        # calculate average training loss end of the epoch
        avg_train_loss = torch.tensor(
            [x["result"]["batch_train_loss"] for x in training_step_outputs]
        ).mean()
        # calculate average training accuracy end of the epoch
        avg_train_acc = torch.tensor(
            [x["result"]["batch_train_acc"] for x in training_step_outputs]
        ).mean()
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log("train_loss", avg_train_loss, prog_bar=True)
        self.log("train_acc", avg_train_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y = batch
        # reduce dimension
        y = torch.squeeze(y)
        # convert to long
        y = y.long()
        # get prediction
        y_pred = self(x)
        # calculate loss
        loss = F.cross_entropy(y_pred, y)
        # get probability score using softmax
        prob = F.softmax(y_pred, dim=1)
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        # calculate accuracy
        acc = torchmetrics.functional.accuracy(pred, y)
        # acc = f1_score(pred, y, num_classes=3)
        dic = {"batch_val_loss": loss, "batch_val_acc": acc}
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log(
            "batch_val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "batch_val_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # return dict
        return dic

    def validation_epoch_end(self, validation_step_outputs):
        # calculate average validation loss end of the epoch
        avg_val_loss = torch.tensor(
            [x["batch_val_loss"] for x in validation_step_outputs]
        ).mean()
        # calculate average validation accuracy end of the epoch
        avg_val_acc = torch.tensor(
            [x["batch_val_acc"] for x in validation_step_outputs]
        ).mean()
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log("val_loss", avg_val_loss, prog_bar=True)
        self.log("val_acc", avg_val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y = batch
        # reduce dimension
        y = torch.squeeze(y)
        # convert to long
        y = y.long()
        # get prediction
        y_pred = self(x)
        # calculate loss
        loss = F.cross_entropy(y_pred, y)
        # get probability score using softmax
        prob = F.softmax(y_pred, dim=1)
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        # calculate accuracy
        acc = torchmetrics.functional.accuracy(pred, y)
        dic = {"batch_test_loss": loss, "batch_test_acc": acc}
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log("batch_test_loss", loss, prog_bar=True)
        self.log("batch_test_acc", acc, prog_bar=True)
        # return dict
        return dic

    def test_epoch_end(self, test_step_outputs):
        # calculate average validation loss end of the epoch
        avg_test_loss = torch.tensor(
            [x["batch_test_loss"] for x in test_step_outputs]
        ).mean()
        # calculate average validation accuracy end of the epoch
        avg_test_acc = torch.tensor(
            [x["batch_test_acc"] for x in test_step_outputs]
        ).mean()
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log("test_loss", avg_test_loss, prog_bar=True)
        self.log("test_acc", avg_test_acc, prog_bar=True)

    def configure_optimizers(self):
        # adam optimiser
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # learning rate reducer scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-15, verbose=True
        )
        # scheduler reduces learning rate based on the value of val_loss metric
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }
