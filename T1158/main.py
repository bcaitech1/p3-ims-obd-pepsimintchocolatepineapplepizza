import sys

sys.path.append("/opt/ml/semantic-segmentation")
import random
from pathlib import Path

import wandb
import fire
import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from T1158.model import get_model
from T1158.dataset import ImageDataset, TestDataset
from T1158.train import train
from T1158.inference import inference
from T1158.slack import alert
from T1158.params import PARAMS
from T1158.misc import FILE_PATHS, get_model_path


def set_seed(random_seed=1818):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    return


def mkdir():
    Path(FILE_PATHS["model_dir"]).mkdir(parents=True, exist_ok=True)
    return


@alert
def main(mode="test"):
    set_seed()
    config = PARAMS["config"]
    model = get_model(config["model"])
    criterion = torch.nn.CrossEntropyLoss()
    params = [
        {"params": model.encoder.parameters(), "lr": config["f-lr"]},
        {"params": model.fcn.parameters(), "lr": config["lr"]},
        {"params": model.classifier.parameters(), "lr": config["lr"]}
    ]
    if hasattr(model, "decoder"):
        params.append({"params": model.decoder.parameters(), "lr": config["lr"]})
    optimizer = torch.optim.Adam(params=params)
    model.cuda()
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])
    if mode == "test":
        wandb.login()
        train_dataset = ImageDataset(FILE_PATHS["train_file"], transform)
        val_dataset = ImageDataset(FILE_PATHS["val_file"], transform)
        train_loader = DataLoader(train_dataset,
                                  batch_size=config["batch_size"],
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=config["num_workers"])
        val_loader = DataLoader(val_dataset,
                                batch_size=config["batch_size"],
                                shuffle=False,
                                pin_memory=True,
                                num_workers=config["num_workers"])
        with wandb.init(**PARAMS) as logger:
            train(model, optimizer, criterion, train_loader, config["epoch"],
                  val_loader, logger)
    elif mode == "inference":
        dataset = ImageDataset(FILE_PATHS["all_train_file"], transform)
        data_loader = DataLoader(dataset,
                                 batch_size=config["batch_size"],
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=config["num_workers"])
        train(model, optimizer, criterion, data_loader, config["epoch"])
        torch.save(model.state_dict(), get_model_path())
        dataset = TestDataset(FILE_PATHS["test_file"], transform)
        data_loader = DataLoader(dataset,
                                 batch_size=config["batch_size"],
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=config["num_workers"])
        inference(model, data_loader)
    else:
        raise ValueError("incorrect mode: %s" % mode)
    return


if __name__ == '__main__':
    mkdir()
    fire.Fire(main)
