import os

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class _ImageDataset(Dataset):
    image_dir = "/opt/ml/input/data"

    def __init__(self, file_path):
        super().__init__()
        self.coco = COCO(file_path)

    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        img = Image.open(os.path.join(self.image_dir, image_infos['file_name']))
        img = np.array(img)
        return img, image_infos

    def __len__(self) -> int:
        return len(self.coco.getImgIds())


class ImageDataset(_ImageDataset):
    def __init__(self, file_path, transform=None):
        super().__init__(file_path)
        self.transform = transform

    def __getitem__(self, index: int):
        img, image_infos = super().__getitem__(index)
        ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((image_infos["height"], image_infos["width"]), dtype=np.int64)
        for ann in anns:
            mask = np.maximum(self.coco.annToMask(ann) * (ann["category_id"] + 1),
                              mask)
        if self.transform is not None:
            return self.transform(image=img, mask=mask)
        return {"image": img, "mask": mask}


class TestDataset(_ImageDataset):
    def __init__(self, file_path, transform=None):
        super().__init__(file_path)
        self.transform = transform

    def __getitem__(self, index: int):
        img, image_infos = super().__getitem__(index)
        if self.transform is not None:
            return self.transform(image=img, file_name=image_infos["file_name"])
        return {"image": img, "file_name": image_infos["file_name"]}
