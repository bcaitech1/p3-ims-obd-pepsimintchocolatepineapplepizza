import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import parmap

from T1158.model import get_model
from T1158.params import PARAMS
from T1158.dataset import TestDataset
from T1158.misc import FILE_PATHS, get_model_path
from T1158.misc import get_result_path


@torch.no_grad()
def inference(model, data_loader):
    image_ids = []
    imgs_list = []
    probs_list = []
    model.eval()
    for inputs in data_loader:
        imgs = inputs.pop("image")
        image_id = inputs.pop("file_name")
        origin_imgs = inputs.pop("mask")
        outputs = model(imgs.cuda())
        probs = torch.nn.functional.softmax(outputs, 1).detach().cpu().numpy()
        origin_imgs = origin_imgs.detach().cpu().numpy()
        probs_list.append(probs)
        imgs_list.append(origin_imgs)
        image_ids.extend(image_id)
    imgs = np.vstack(imgs_list)
    probs = np.vstack(probs_list)
    crf_outputs = parmap.starmap_async(dense_crf, zip(image_ids, imgs, probs),
                                       pm_processes=4).get()
    result = pd.DataFrame(columns=["image_id", "PredictionString"])
    idx = 0
    for image_id, crf in crf_outputs:
        crf = crf.argmax(0).flatten().tolist()
        result.loc[idx] = [image_id, " ".join(map(str, crf))]
        idx += 1
    result.to_csv(get_result_path(), index=False)
    return


def inference_with_saved_model():
    config = PARAMS["config"]
    model = get_model(config["model"])
    model.load_state_dict(torch.load(get_model_path()))
    model.cuda()
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])
    dataset = TestDataset(FILE_PATHS["test_file"], transform)
    data_loader = DataLoader(dataset,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             pin_memory=True,
                             num_workers=2)
    inference(model, data_loader)
    return


MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3


def dense_crf(image_id, img, prob):
    """https://github.com/zllrunning/deeplab-pytorch-crf"""
    c, h, w = prob.shape
    U = utils.unary_from_softmax(prob)
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)
    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)
    Q = d.inference(MAX_ITER)
    return image_id, np.array(Q).reshape((c, h, w))


if __name__ == '__main__':
    inference_with_saved_model()
