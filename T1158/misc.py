import os

from T1158.params import PARAMS

FILE_PATHS = {
    "train_file": "/opt/ml/input/data/train.json",
    "val_file": "/opt/ml/input/data/val.json",
    "all_train_file": "/opt/ml/input/data/train_all.json",
    "test_file": "/opt/ml/input/data/test.json",
    "result_dir": "/opt/ml/semantic-segmentation/T1158/result",
    "model_dir": "/opt/ml/semantic-segmentation/T1158/result/model"
}


def get_model_path(model_name=None, version=None):
    if model_name is None:
        model_name = PARAMS["config"]["model"]
    if version is None:
        version = PARAMS["job_type"]
    return os.path.join(
        FILE_PATHS["model_dir"], "{}-{}.pth".format(model_name, version)
    )


def get_result_path():
    file_name = PARAMS["config"]["model"]
    version = PARAMS["job_type"]
    return os.path.join(FILE_PATHS["result_dir"],
                        "{}-{}.csv".format(file_name, version))
