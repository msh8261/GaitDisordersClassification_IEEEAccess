# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
import yaml
from sklearn.model_selection import KFold

from nn.classification.cnn import CNN, CNN2D
from nn.classification.mlp import MLP, MLP_new
from nn.classification.rnns import GRU, LSTM, RNN, RNN_1, LSTM_new, RNN_new
from nn.tools.adabound import AdaBound
from src.dataset import GaitData
# import config.config_train as config
from src.load import LoadData

with open("config/config_train.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes_ = config["input_data_params"]["num_class"]
sequence_length = config["input_data_params"]["sequences"]


weights = torch.tensor([1.0, 2.0, 2.0]).to(device)
# weights = torch.tensor([0.4, 0.77, 0.8]).to(device)


def matplotlib_imshow(img, one_channel=False):
    img = img.detach().cpu()
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class GaitModel(pl.LightningModule):
    def __init__(
        self,
        k,
        random_state,
        X_train_path,
        y_train_path,
        X_test_path,
        y_test_path,
        input_size,
        model_name,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.k = k
        self.num_splits = config["train_params"]["n_folds"]
        self.split_seed = random_state

        self.batch_size = config["train_params"]["batch_size"]
        sequence_length = config["input_data_params"]["sequences"]
        num_class = config["input_data_params"]["num_class"]
        hidden_size = input_size * 1
        num_layers = 1

        self.X_train_path = X_train_path
        self.y_train_path = y_train_path
        self.X_test_path = X_test_path
        self.y_test_path = y_test_path

        ld_tr = LoadData(
            X_train_path,
            y_train_path,
            config["input_data_params"]["num_augmentation"],
            True,
        )
        # for test dataset augmentation shoud be set to 0
        ld_ts = LoadData(X_test_path, y_test_path, 0, False)

        self.X_train = ld_tr.get_X()
        self.y_train = ld_tr.get_y()

        self.X_test = ld_ts.get_X()
        self.y_test = ld_ts.get_y()

        self.model_name = model_name

        if model_name == "lstm":
            model = LSTM(input_size, hidden_size, num_layers, num_class)
        elif model_name == "lstm_new":
            model = LSTM_new(input_size, hidden_size, num_layers, num_class)
        elif model_name == "gru":
            model = GRU(input_size, hidden_size, num_layers, num_class)
        elif model_name == "cnn":
            model = CNN(input_size, hidden_size, num_layers, num_class)
        elif model_name == "cnn2d":
            model = CNN2D(input_size, hidden_size, num_layers, num_class)
        elif model_name == "mlp":
            model = MLP(input_size, hidden_size, num_layers, num_class)
        elif model_name == "mlp_new":
            model = MLP_new(input_size, hidden_size, num_layers, num_class)
        elif model_name == "rnn":
            model = RNN(input_size, hidden_size, num_layers, num_class)
        elif model_name == "rnn_new":
            model = RNN_new(input_size, hidden_size, num_layers, num_class)
        elif model_name == "rnn_1":
            model = RNN_1(input_size, hidden_size, num_layers, num_class)
        elif model_name == "vit1D":
            custom_config = {
                "in_chans": sequence_length,
                "embed_dim": input_size,
                "n_classes": num_class,
                "depth": 2,  # number of TF blocks
                "n_heads": 2,
                "qkv_bias": True,
                "mlp_ratio": 4,
            }
            model = vit1D(**custom_config)
        else:
            raise ValueError(f"{model_name} is nkonw model name")

        self.model = model

        lr = float(config["train_params"]["learning_rate"])
        weight_decay = float(config["train_params"]["weight_decay"])

        optimizers = {
            "Adam": torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            ),
            "SGD": torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
            ),
            "RMSprop": torch.optim.RMSprop(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            ),
            "Adadelta": torch.optim.Adadelta(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            ),
            "Adagrad": torch.optim.Adagrad(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            ),
            "Adamax": torch.optim.Adamax(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            ),
            "Adamw": torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            ),
            "AdaBound": AdaBound(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            ),
        }

        self.optimizer = optimizers[config["train_params"]["opt_indx"]]

        self.critrion = nn.CrossEntropyLoss(weight=weights)
        # alpha = 1.0, gamma = 2.0
        alpha = 1.0
        gamma = 2.0
        # self.critrion = FocalLoss(alpha, gamma)

    def forward(self, x):
        return self.model(x)

    def metrics(self, pred, y, num_classes):
        self.acc = torchmetrics.functional.accuracy(pred, y)
        self.f1_score = torchmetrics.functional.f1(
            pred, y, num_classes=num_classes_, average="weighted"
        )
        self.precision = torchmetrics.functional.precision(
            pred, y, num_classes=num_classes_, average="weighted"
        )
        self.recall = torchmetrics.functional.recall(
            pred, y, num_classes=num_classes_, average="weighted"
        )
        self.auc = torchmetrics.functional.auc(pred, y, reorder=True)
        self.specificity = torchmetrics.functional.specificity(
            pred, y, num_classes=num_classes
        )
        self.confmat = torchmetrics.functional.confusion_matrix(
            pred, y, num_classes=num_classes
        )
        return self

    def setup(self, stage=None):
        # for kfold method
        kf = KFold(n_splits=self.num_splits, random_state=self.split_seed, shuffle=True)
        all_splits = [k for k in kf.split(self.X_train)]
        train_indexes, val_indexes = all_splits[self.k]
        train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

        self.train_set = GaitData(
            self.X_train[train_indexes], self.y_train[train_indexes]
        )
        self.val_set = GaitData(self.X_train[val_indexes], self.y_train[val_indexes])
        #########################################################
        ## if no need kfold method
        # train_percent = 0.7
        # test_percent = 0
        # val_percent = 1 - (train_percent + test_percent)

        # train_size = int(train_percent * len(self.X_train))
        # val_size = int(val_percent * (len(self.X_train)))
        # test_size = int(len(self.X_train) - (train_size + val_size))

        # data_train_val = GaitData(self.X_train, self.y_train)
        # self.train_set, self.val_set, _ = torch.utils.data.random_split(data_train_val, (train_size, val_size, test_size))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        data_test = GaitData(self.X_test, self.y_test)
        return torch.utils.data.DataLoader(
            data_test, batch_size=self.batch_size, shuffle=False
        )

    # @property
    # def automatic_optimization(self):
    #     return False

    def set_data_get_data(self, batch):
        x, y = batch
        y = torch.squeeze(y)
        y = y.long()
        return x, y

    def training_step(self, batch, batch_nb):
        if batch_nb == 0:
            x, y = batch
            self.x_samples = x
            self.reference_image = (batch[0][0]).unsqueeze(0)
            # self.reference_image.resize((1,1,300,104))
            print(self.reference_image.shape)

        x, y = self.set_data_get_data(batch)

        cls = self(x)

        cls = F.softmax(cls, dim=1)
        preds = cls.data.max(dim=1)[1]

        loss = self.critrion(cls, y)

        # loss.backward()
        # self.optimizer.step()

        acc = torchmetrics.functional.accuracy(preds, y)

        cl = self.metrics(preds, y, num_classes_)
        acc = cl.acc
        f1_score = cl.f1_score
        precision = cl.precision
        recall = cl.recall
        auc = cl.auc
        specificity = cl.specificity
        confmat = cl.confmat

        correct = cls.argmax(dim=1).eq(y).sum().item()
        total = len(y)

        dic = {
            "batch_train_loss": loss,
            "batch_train_acc": acc,
            "batch_train_f1": f1_score,
            "batch_train_precision": precision,
            "batch_train_recall": recall,
            "correct": correct,
            "total": total,
        }
        self.log("batch_train_loss", loss, prog_bar=True)
        self.log("batch_train_acc", acc, prog_bar=True)
        self.log("batch_train_f1", f1_score, prog_bar=True)
        self.log("batch_train_precision", precision, prog_bar=True)
        self.log("batch_train_recall", recall, prog_bar=True)
        return {"loss": loss, "result": dic}

    def training_epoch_end(self, train_step_output):
        ave_loss = torch.tensor(
            [x["result"]["batch_train_loss"] for x in train_step_output]
        ).mean()
        ave_acc = torch.tensor(
            [x["result"]["batch_train_acc"] for x in train_step_output]
        ).mean()
        avg_train_f1 = torch.tensor(
            [x["result"]["batch_train_f1"] for x in train_step_output]
        ).mean()
        avg_train_precision = torch.tensor(
            [x["result"]["batch_train_precision"] for x in train_step_output]
        ).mean()
        avg_train_recall = torch.tensor(
            [x["result"]["batch_train_recall"] for x in train_step_output]
        ).mean()
        self.log("average_train_loss", ave_loss, prog_bar=True)
        self.log("average_train_acc", ave_acc, prog_bar=True)
        self.log("average_train_f1", avg_train_f1, prog_bar=True)
        self.log("average_train_precision", avg_train_precision, prog_bar=True)
        self.log("average_train_recall", avg_train_recall, prog_bar=True)

        avg_loss = torch.stack([x["loss"] for x in train_step_output]).mean()
        print("Loss train= {}".format(avg_loss))
        correct = sum([x["result"]["correct"] for x in train_step_output])
        total = sum([x["result"]["total"] for x in train_step_output])
        # tensorboard_logs = {'loss': avg_loss,"Accuracy": correct/total}

        # Loggig scalars
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            "Accuracy/Train", correct / total, self.current_epoch
        )

        # # add images for tesorboard visualization
        # self.add_image(self.x_samples)

        # Logging histograms
        # self.custom_histogram_adder()

        # self.custom_heatmap_adder(self.confmat, num_classes_)

        print("Confusion matrix: ", self.confmat)
        print(
            "Number of Correctly identified Training Set Images {} from a set of {}. \nAccuracy= {} ".format(
                correct, total, correct / total
            )
        )

    def validation_step(self, batch, batch_nb):
        x, y = self.set_data_get_data(batch)

        cls = self(x)

        cls = F.softmax(cls, dim=1)
        preds = cls.data.max(dim=1)[1]

        loss = self.critrion(cls, y)

        cl = self.metrics(preds, y, num_classes_)
        acc = cl.acc
        f1_score = cl.f1_score
        precision = cl.precision
        recall = cl.recall
        auc = cl.auc
        specificity = cl.specificity
        confmat = cl.confmat

        dic = {
            "batch_val_loss": loss,
            "batch_val_acc": acc,
            "batch_val_f1": f1_score,
            "batch_val_precision": precision,
            "batch_val_recall": recall,
        }
        self.log("batch_train_loss", loss, prog_bar=True)
        self.log("batch_train_acc", acc, prog_bar=True)
        self.log("batch_val_f1", f1_score, prog_bar=True, logger=True)
        self.log("batch_val_precision", precision, prog_bar=True, logger=True)
        self.log("batch_val_recall", recall, prog_bar=True, logger=True)

        return dic

    def validation_epoch_end(self, val_step_output):
        ave_loss = torch.tensor([x["batch_val_loss"] for x in val_step_output]).mean()
        ave_acc = torch.tensor([x["batch_val_acc"] for x in val_step_output]).mean()
        avg_val_f1 = torch.tensor([x["batch_val_f1"] for x in val_step_output]).mean()
        avg_val_precision = torch.tensor(
            [x["batch_val_precision"] for x in val_step_output]
        ).mean()
        avg_val_recall = torch.tensor(
            [x["batch_val_recall"] for x in val_step_output]
        ).mean()
        self.log("average_val_loss", ave_loss, prog_bar=True)
        self.log("average_val_acc", ave_acc, prog_bar=True)
        self.log("average_val_f1", avg_val_f1, prog_bar=True)
        self.log("average_val_precision", avg_val_precision, prog_bar=True)
        self.log("average_val_recall", avg_val_recall, prog_bar=True)

    def test_step(self, batch, batch_np):
        x, y = self.set_data_get_data(batch)

        cls = self(x)

        cls = F.softmax(cls, dim=1)
        preds = cls.data.max(dim=1)[1]

        cl = self.metrics(preds, y, num_classes_)
        acc = cl.acc
        f1_score = cl.f1_score

        dic = {"batch_test_acc": acc, "batch_test_f1": f1_score}
        self.log("batch_test_acc", acc, prog_bar=True)
        self.log("batch_test_f1", f1_score, prog_bar=True)
        return dic

    def test_epoch_end(self, test_step_output):
        ave_acc = torch.tensor([x["batch_test_acc"] for x in test_step_output]).mean()
        ave_f1 = torch.tensor([x["batch_test_f1"] for x in test_step_output]).mean()
        self.log("average_test_acc", ave_acc, prog_bar=True)
        self.log("average_test_f1", ave_f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optimizer
        factor_ = 0.5
        patience_ = 50
        min_lr_ = 1e-15
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor_,
            patience=patience_,
            min_lr=min_lr_,
            verbose=True,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] *= config["train_params"]["lr_decay"]
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "average_val_loss",
        }

    def custom_heatmap_adder(self, confusion_matrix, num_classes):
        df_cm = pd.DataFrame(
            confusion_matrix.detach().cpu().numpy(),
            index=range(num_classes),
            columns=range(num_classes),
        )
        fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def custom_histogram_adder(self):
        # A custom defined function that adds Histogram to TensorBoard
        # Iterating over all parameters and logging them
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def add_image(self, x):
        writer = torch.utils.tensorboard.SummaryWriter("lightning_logs/")
        # create grid of images
        img_grid = torchvision.utils.make_grid(x)
        # show images
        matplotlib_imshow(img_grid, one_channel=True)
        # write to tensorboard
        writer.add_image("sample_images", img_grid)
        writer.close()
