import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
from warmup_scheduler import GradualWarmupScheduler

CLASSES = 12


def get_scheduler(optimizer, name):
    if name == "gradual_warmup":
        exp_lr = ExponentialLR(optimizer, gamma=0.996)
        return GradualWarmupScheduler(optimizer,
                                      multiplier=1,
                                      total_epoch=800,
                                      after_scheduler=exp_lr)
    if name == "cosine":
        return CosineAnnealingWarmRestarts(optimizer, 100, 2)
    raise ValueError("incorrect scheduler name: %s" % name)


def train(model, optimizer, criterion, train_loader, epoch,
          val_loader=None, logger=None):
    step = 1
    # scheduler = get_scheduler(optimizer, "cosine")
    for i in range(epoch):
        for inputs in train_loader:
            model.train()
            optimizer.zero_grad()
            inputs = {key: val.cuda() for key, val in inputs.items()}
            masks = inputs.pop("mask").long()
            outputs = model(inputs["image"])
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            if logger is not None and step % 20 == 0:
                label_trues = masks.detach().cpu().numpy()
                label_preds = outputs.argmax(1).detach().cpu().numpy()
                acc, _, mean_iu = label_accuracy_score(label_trues, label_preds)
                val_loss, val_acc, _, val_mean_iu = evaluate(model, criterion, val_loader)
                logger.log({
                    "epoch": i,
                    "train_loss": loss.item(),
                    "train_acc": acc,
                    "train_miou": mean_iu,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_miou": val_mean_iu,
                    # "lr": optimizer.param_groups[0]['lr']
                }, step=step)
            step += 1
    return


def get_hist(label_true, label_pred):
    mask = (label_true >= 0) & (label_true < CLASSES)  # filtering incorrect label
    return np.bincount(
        CLASSES * label_true[mask].astype(int) + label_pred[mask],
        minlength=CLASSES ** 2).reshape(CLASSES, CLASSES)


def get_metric(hist):
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_precision = np.diag(hist) / hist.sum(axis=1)
    mean_precision = np.nanmean(mean_precision)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    return acc, mean_precision, mean_iu


def label_accuracy_score(label_trues, label_preds):
    acc, mean_precision, mean_iu = 0, 0, 0
    length = len(label_trues)
    for lt, lp in zip(label_trues, label_preds):
        hist = get_hist(lt.flatten(), lp.flatten())
        _acc, _mean_precision, _mean_iu = get_metric(hist)
        acc += _acc
        mean_precision += _mean_precision
        mean_iu += _mean_iu
    return acc / length, mean_precision / length, mean_iu / length


@torch.no_grad()
def evaluate(model, criterion, val_loader):
    val_loss = 0
    acc, mean_precision, mean_iu = 0, 0, 0
    step = 0
    model.eval()
    for inputs in val_loader:
        inputs = {key: val.cuda() for key, val in inputs.items()}
        masks = inputs.pop("mask").long()
        outputs = model(inputs["image"])
        loss = criterion(outputs, masks)
        val_loss += loss.item() * len(masks)
        label_preds = outputs.argmax(1).detach().cpu().numpy()
        label_trues = masks.detach().cpu().numpy()
        _acc, _mean_precision, _mean_iu = label_accuracy_score(label_trues, label_preds)
        length = len(masks)
        acc += _acc * length
        mean_precision += _mean_precision * length
        mean_iu += _mean_iu * length
        step += length
    return val_loss / step, acc / step, mean_precision / step, mean_iu / step
