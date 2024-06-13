# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103, e1101
import os
# import glob
# import numpy as np
# import math
import shutil
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
# from sklearn import metrics
import yaml
from pytorch_lightning.callbacks import EarlyStopping

# import config.config_train as config
from nn.classification.classification import GaitModel
from prepare_dataset.tools.decorators import logger

# import matplotlib.pyplot as plt


# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks import LearningRateMonitor
# from pytorch_lightning.loggers import TensorBoardLogger


# import torch.nn as nn
# from torch.autograd import Variable
# import torch.nn.functional as F


# from prepare_dataset.tools.decorators import timeit, countcall, dataclass

# from src.dataset import GaitData
# from src.load import LoadData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    k,
    random_state,
    X_train_path,
    y_train_path,
    X_test_path,
    y_test_path,
    input_size,
    input_size_str,
    model_name,
    method,
    mode,
):
    pl.seed_everything(random_state)
    parser = ArgumentParser()
    args, unknown = parser.parse_known_args()
    parent_parser = pl.Trainer.add_argparse_args(parser)
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    model = GaitModel(
        k,
        random_state,
        X_train_path,
        y_train_path,
        X_test_path,
        y_test_path,
        input_size,
        model_name,
    )

    if method is not None:
        model_name = f"{model_name}_{method}"

    dir_checkpoint = f"checkpoints/{mode}/{model_name}_{input_size_str}"
    os.makedirs(dir_checkpoint, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=dir_checkpoint,
        filename=f"best-checkpoint_{model_name}_{input_size_str}",
        save_top_k=1,
        verbose=True,
        monitor="average_val_loss",
        mode="min",
    )

    logger = pl.loggers.TensorBoardLogger(
        f"lightning_logs/{mode}", name=f"model_run_{model_name}_{input_size_str}"
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=config["train_params"]["epochs"],
        deterministic=True,
        gpus=1,
        progress_bar_refresh_rate=1,
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="average_val_loss", patience=50),
            checkpoint_callback,
            lr_monitor,
        ],
    )

    tuner = pl.tuner.tuning.Tuner(trainer)

    trainer.fit(model)

    trainer.test(model)

    return model, trainer


def save_scripted_module(model, save_model_path):
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, save_model_path)


def save_traced_module(model, save_model_path, input_size):
    traced_model = torch.jit.trace(
        model,
        torch.rand(1, sequence_length, input_size, dtype=torch.float32, device="cuda"),
    )
    torch.jit.save(traced_model, save_model_path)


def convert_ckp_to_pt_model(
    best_k,
    best_r,
    path_model,
    X_train_path,
    y_train_path,
    X_test_path,
    y_test_path,
    model_name,
    mode,
    input_size,
    input_size_str,
):
    pl.seed_everything(best_r)
    model = GaitModel(
        best_k,
        best_r,
        X_train_path,
        y_train_path,
        X_test_path,
        y_test_path,
        input_size,
        model_name,
    )
    model.to(device)
    model.eval()
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint["state_dict"])
    save_model_path = f"saved_models/{mode}/{model_name}_{input_size_str}.pt"

    # if model_name == 'lstm' or model_name == 'gru': # or model_name == 'transformer':
    #     save_traced_module(model, save_model_path)
    # else:
    #     save_scripted_module(model, save_model_path)
    save_scripted_module(model, save_model_path)


def save_metrics_to_file(
    best_k,
    best_r,
    model_name,
    model_scores,
    nums_folds,
    num_classes,
    method,
    mode,
    input_size,
):
    if method is not None:
        base_dir = f"results/{mode}/{model_name}_{method}_{input_size}"
        os.makedirs(base_dir, exist_ok=True)
        file_name = f"{base_dir}/{str(model_name)}_{method}_{input_size}_fold{best_k}_random{best_r}_{str(num_classes)}classes_results.txt"
        file_name_check = f"{base_dir}/{str(model_name)}_{method}_{input_size}_fold{best_k}_random{best_r}_{str(num_classes)}classes_results_check.txt"
    else:
        base_dir = f"results/{mode}/{model_name}_{input_size}"
        os.makedirs(base_dir, exist_ok=True)
        file_name = f"{base_dir}/{str(model_name)}_{input_size}_fold{best_k}_random{best_r}_{str(num_classes)}classes_results.txt"
        file_name_check = f"{base_dir}/{str(model_name)}_{input_size}_fold{best_k}_random{best_r}_{str(num_classes)}classes_results_check.txt"
    with open(file_name_check, "w") as f:
        for j, score in enumerate(model_scores):
            f.write("\n")
            for key, value in score:
                f.write(f"{key}: {value.cpu().detach().numpy()}, ")
            f.write(
                "\n======================================================================="
            )
    with open(file_name, "w") as f:
        for j, score in enumerate(model_scores):
            for key, value in score:
                if j == 0:
                    f.write("%s," % (key))
            f.write("\n")
            for key, value in score:
                f.write("%s," % (value.cpu().detach().numpy()))


def show_metrics(trainer, model_name):
    print("===================================================================")
    metrics = trainer.callback_metrics
    print(f'average_train_loss: {metrics["average_train_loss"]:.2f}')
    print(f'average_train_acc: {metrics["average_train_acc"]*100:.2f} %')
    print(f'average_train_f1: {metrics["average_train_f1"]*100:.2f} %')
    print(f'average_val_loss: {metrics["average_val_loss"]:.2f}')
    print(f'average_val_acc: {metrics["average_val_acc"]*100:.2f} %')
    print(f'average_val_f1: {metrics["average_val_f1"]*100:.2f} %')
    print(f'average_test_acc: {metrics["average_test_acc"]*100:.2f} %')
    print(f'average_test_f1: {metrics["average_test_f1"]*100:.2f} %')
    return metrics.items(), metrics["average_test_f1"], metrics["average_test_acc"]


@logger
def delete_prev_models(dir_path, model_name, indx, method):
    cnt = 0
    files = os.listdir(dir_path)
    if files:
        for file_name in files:
            # construct full file path
            file = os.path.join(dir_path, file_name)
            check_name = file_name.split("_")
            current_name = check_name[indx]
            if method is not None:
                current_method = check_name[indx + 1]
                if f"{model_name}" in check_name and f"{method}" in check_name:
                    cnt += 1
                if (
                    os.path.isfile(file)
                    and cnt > 0
                    and model_name == current_name
                    and method == current_method
                ):
                    print("Deleting file:", file)
                    os.remove(file)
            elif method is None:
                if f"{model_name}" in check_name:
                    cnt += 1
                if os.path.isfile(file) and cnt > 0 and model_name == current_name:
                    print("Deleting file:", file)
                    os.remove(file)
            else:
                assert "INFO: model_name or method name problem."


def do_train(train_mode, train_dataset_path, input_size, mode, best_feats, kind_62):
    if input_size == 62:
        input_size_str = f"{input_size}_{kind_62}"
    else:
        input_size_str = f"{input_size}"
    if train_mode:
        for j, model_name in enumerate(models_name):
            print("==========================================")
            print(f"train for {model_name} model")
            print("==========================================")
            best_r = 0
            for i, r in enumerate(random_state_list):
                if best_feats:
                    for method in bfs_methods:
                        dir_path = f"saved_models/{mode}/{model_name}_{method}_{input_size_str}"
                        os.makedirs(dir_path, exist_ok=True)
                        model_scores = []
                        best_score = 0
                        best_k = 0
                        number_final_models = len(models_name) * len(bfs_methods)
                        os.makedirs(
                            f"./lightning_logs/{mode}/model_run_{model_name}_{method}_{input_size_str}",
                            exist_ok=True,
                        )
                        bfs_method = f"{method}_{input_size_str}_selected_best_feats"
                        X_train_path = os.path.join(
                            train_dataset_path, f"Xtrain_{bfs_method}.File"
                        )
                        y_train_path = os.path.join(train_dataset_path, "ytrain.File")
                        X_test_path = os.path.join(
                            train_dataset_path, f"Xtest_{bfs_method}.File"
                        )
                        y_test_path = os.path.join(train_dataset_path, "ytest.File")
                        for k in range(n_folds):
                            path_model = os.path.join(
                                dir_path,
                                f"{model_name}_{method}_{input_size_str}_k{k}_rand{r}.pt",
                            )
                            checkpoint_dir = f"./checkpoints/{mode}/{model_name}_{method}_{input_size_str}"
                            if os.path.exists(checkpoint_dir):
                                delete_prev_models(
                                    f"./checkpoints/{mode}/{model_name}_{method}_{input_size_str}",
                                    model_name,
                                    1,
                                    method,
                                )
                            shutil.rmtree(
                                f"./lightning_logs/{mode}/model_run_{model_name}_{method}_{input_size_str}"
                            )
                            model, trainer = train(
                                k,
                                r,
                                X_train_path,
                                y_train_path,
                                X_test_path,
                                y_test_path,
                                input_size,
                                input_size_str,
                                model_name,
                                method,
                                mode,
                            )
                            metrics_items, test_f1, test_acc = show_metrics(
                                trainer, model_name
                            )
                            model_scores.append(metrics_items)
                            if best_score < (test_f1 + test_acc) / 2:
                                if os.path.exists(dir_path):
                                    delete_prev_models(dir_path, model_name, 0, method)
                                save_scripted_module(model, path_model)
                                best_score = test_f1
                                best_k = k
                                best_r = r
                        save_metrics_to_file(
                            best_k,
                            best_r,
                            model_name,
                            model_scores,
                            n_folds,
                            num_classes,
                            method,
                            mode,
                            input_size,
                        )
                else:
                    dir_path = f"saved_models/{mode}/{model_name}_{input_size_str}"
                    os.makedirs(dir_path, exist_ok=True)
                    model_scores = []
                    best_score = 0
                    best_k = 0
                    method = None
                    number_final_models = len(models_name)
                    os.makedirs(
                        f"./lightning_logs/{mode}/model_run_{model_name}_{input_size_str}",
                        exist_ok=True,
                    )
                    X_train_path = os.path.join(train_dataset_path, f"Xtrain.File")
                    y_train_path = os.path.join(train_dataset_path, "ytrain.File")
                    X_test_path = os.path.join(train_dataset_path, f"Xtest.File")
                    y_test_path = os.path.join(train_dataset_path, "ytest.File")
                    for k in range(n_folds):
                        path_model = os.path.join(
                            dir_path, f"{model_name}_{input_size_str}_k{k}_rand{r}.pt"
                        )
                        checkpoint_dir = (
                            f"./checkpoints/{mode}/{model_name}_{input_size_str}"
                        )
                        if os.path.exists(checkpoint_dir):
                            delete_prev_models(checkpoint_dir, model_name, 1, method)
                        shutil.rmtree(
                            f"./lightning_logs/{mode}/model_run_{model_name}_{input_size_str}"
                        )
                        model, trainer = train(
                            k,
                            r,
                            X_train_path,
                            y_train_path,
                            X_test_path,
                            y_test_path,
                            input_size,
                            input_size_str,
                            model_name,
                            method,
                            mode,
                        )
                        metrics_items, test_f1, test_acc = show_metrics(
                            trainer, model_name
                        )
                        model_scores.append(metrics_items)
                        if best_score < (test_f1 + test_acc) / 2:
                            if os.path.exists(dir_path):
                                delete_prev_models(dir_path, model_name, 0, method)
                            save_scripted_module(model, path_model)
                            best_score = test_f1
                            best_k = k
                            best_r = r

                    save_metrics_to_file(
                        best_k,
                        best_r,
                        model_name,
                        model_scores,
                        n_folds,
                        num_classes,
                        method,
                        mode,
                        input_size_str,
                    )

    else:
        raise ValueError("Please specified train_mode=True")


if __name__ == "__main__":
    with open("config/config_train.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    n_folds = config["train_params"]["n_folds"]
    random_state_list = config["random_state_list"]
    models_name = config["model_params"]["models_name"]
    num_classes = config["input_data_params"]["num_class"]
    sequence_length = config["input_data_params"]["sequences"]
    bfs_methods = config["model_params"]["bfs_methods"]

    modes = config["model_params"]["modes"]
    print("@@@@@@@@@@@@", modes)
    inputs_best_feat_size = config["input_data_params"]["inputs_best_feat_size"]
    inputs_manual_feat_size = config["input_data_params"]["inputs_manual_feat_size"]

    train_mode = True
    for mode in modes:
        select_kind_62 = None
        # run to train on best features selected
        # inputs_best_feat_size = [60, 50, 40, 30, 20, 10]
        for input_best_feat_size in inputs_best_feat_size:
            best_feat_mode = True
            train_dataset_path = f"./data/{mode}/best_feats/final_{input_best_feat_size}_best_feats_selections_from_70"
            do_train(
                train_mode,
                train_dataset_path,
                input_best_feat_size,
                mode,
                best_feat_mode,
                select_kind_62,
            )

        # run to train on manual features generated
        # inputs_manual_feat_size = [70, 34, 36, 58, 62]
        for input_feat_size in inputs_manual_feat_size:
            best_feat_mode = False
            if input_feat_size == 62:
                select_kinds_62 = ["rsa", "rsd", "rba"]
                for select_kind_62 in select_kinds_62:
                    train_dataset_path = f"./data/{mode}/{input_feat_size}_{select_kind_62}_zero_padding_no"
                    do_train(
                        train_mode,
                        train_dataset_path,
                        input_feat_size,
                        mode,
                        best_feat_mode,
                        select_kind_62,
                    )
            else:
                train_dataset_path = f"./data/{mode}/{input_feat_size}_zero_padding_no"
                do_train(
                    train_mode,
                    train_dataset_path,
                    input_feat_size,
                    mode,
                    best_feat_mode,
                    select_kind_62,
                )
