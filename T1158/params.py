PARAMS = {
    "project": "semantic-segmentation",
    "group": "DeepLabV1VGG16",
    "job_type": "v1",
    "config": {
        "model": "DeepLabV1VGG16",
        "epoch": 100,
        "batch_size": 64,
        "lr": 3e-3,
        "f-lr": 3e-3,
        "criterion": "ce",
        "optimizer": "adam",
        "CRF": True,
        # "scheduler": "cosine",
        "num_workers": 4
    }
}
