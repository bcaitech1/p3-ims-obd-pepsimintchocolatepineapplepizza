import random

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
from T1158.params import PARAMS, FILE_PATHS


def fix_random_state(random_seed=1818):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    return


@alert
def main(mode="test"):
    fix_random_state()
    config = PARAMS["config"]
    model = get_model(config["model"])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=[
        {"params": model.encoder.parameters(), "lr": config["f-lr"]},
        {"params": model.fcn.parameters(), "lr": config["lr"]},
        {"params": model.decoder.parameters(), "lr": config["lr"]}
    ])
    model.cuda()
    transform = A.Compose([
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
            train(model, optimizer, criterion, config["epoch"], train_loader,
                  val_loader, logger)
    elif mode == "inference":
        dataset = ImageDataset(FILE_PATHS["all_train_file"], transform)
        data_loader = DataLoader(dataset,
                                 batch_size=config["batch_size"],
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=config["num_workers"])
        train(model, optimizer, criterion, config["epoch"], data_loader)
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
    fire.Fire(main)
