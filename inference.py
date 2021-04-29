import os
from pathlib import Path

import pandas as pd
import torch
import albumentations as A

from params import PARAMS


@torch.no_grad()
def inference(model, data_loader):
    save_path = "/opt/ml/semantic-segmentation/result"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    size = 256
    resize = A.Compose([A.Resize(size, size)])
    image_ids = []
    preds = []
    model.eval()
    for inputs in data_loader:
        imgs = inputs.pop("image")
        image_id = inputs.pop("file_name")
        outputs = model(imgs.cuda())
        imgs = imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)
        masks = outputs.argmax(1).detach().cpu().numpy()
        for img, mask in zip(imgs, masks):
            transformed = resize(image=img, mask=mask)
            preds.append(transformed['mask'].reshape(1, -1).tolist())
        image_ids.extend(image_id)
    result = pd.DataFrame(columns=["image_id", "PredictionString"])
    idx = 0
    for image_id, pred in zip(image_ids, preds):
        result.loc[idx] = [image_id, " ".join(map(str, pred[0]))]
        idx += 1
    file_name = PARAMS["config"]["model"]
    version = PARAMS["job_type"]
    save_path = os.path.join(save_path, "%s-%s.csv" % (file_name, version))
    result.to_csv(save_path, index=False)
    return
