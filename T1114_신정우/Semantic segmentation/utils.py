# # https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
# import numpy as np


# def _fast_hist(label_true, label_pred, n_class):
#     mask = (label_true >= 0) & (label_true < n_class)
#     hist = np.bincount(
#         n_class * label_true[mask].astype(int) +
#         label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
#     return hist


# def label_accuracy_score(label_trues, label_preds, n_class):
#     """Returns accuracy score evaluation result.
#       - overall accuracy
#       - mean accuracy
#       - mean IU
#       - fwavacc
#     """
#     hist = np.zeros((n_class, n_class))
#     for lt, lp in zip(label_trues, label_preds):
#         hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
#     acc = np.diag(hist).sum() / hist.sum()
#     with np.errstate(divide='ignore', invalid='ignore'):
#         acc_cls = np.diag(hist) / hist.sum(axis=1)
#     acc_cls = np.nanmean(acc_cls)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         iu = np.diag(hist) / (
#             hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
#         )
#     mean_iu = np.nanmean(iu)
#     freq = hist.sum(axis=1) / hist.sum()
#     fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#     return acc, acc_cls, mean_iu, fwavacc

import numpy as np
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    mIoUs = []
    for lt, lp in zip(label_trues, label_preds):
        hist = _fast_hist(lt.flatten(), lp.flatten(), n_class)
        with np.errstate(divide='ignore', invalid='ignore'):
            iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
            )
        mean_iu = np.nanmean(iu)
        mIoUs.append(mean_iu)
    return mIoUs

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-4):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = F.one_hot(targets, num_classes = 12).permute(0,3,1,2).reshape(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs) 
    
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = F.one_hot(targets, num_classes = 12).permute(0,3,1,2).reshape(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice