PARAMS = {
    "project": "semantic-segmentation",
    "group": "UNet4VGG16",
    "job_type": "v1",
    "config": {
        "model": "UNet4VGG16",
        "epoch": 20,
        "batch_size": 16,
        "lr": 3e-4,
        "f-lr": 3e-5,
        "criterion": "ce",
        "optimizer": "adam",
        "num_workers": 4,
    }
}

FILE_PATHS = {
    "train_file": "/opt/ml/input/data/train.json",
    "val_file": "/opt/ml/input/data/val.json",
    "all_train_file": "/opt/ml/input/data/train_all.json",
    "test_file": "/opt/ml/input/data/test.json"
}
