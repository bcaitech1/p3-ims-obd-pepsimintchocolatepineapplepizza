from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)

classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
        "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
# config file 들고오기
cfg = Config.fromfile('/opt/ml/code/mmdetection_trash/configs/efficientdet/efficient_net.py')

PREFIX = '../../input/data/'
cfg.runner.max_epochs = 60



# dataset 바꾸기
cfg.data.train.classes = classes
cfg.data.train.img_prefix = PREFIX
cfg.data.train.ann_file = PREFIX + 'train.json'
cfg.data.train.pipeline[2]['img_scale'] = (512, 512)

cfg.data.val.classes = classes
cfg.data.val.img_prefix = PREFIX
cfg.data.val.ann_file = PREFIX + 'val.json'
cfg.data.val.pipeline[1]['img_scale'] = (512, 512)

cfg.data.test.classes = classes
cfg.data.test.img_prefix = PREFIX
cfg.data.test.ann_file = PREFIX + 'test.json'
cfg.data.test.pipeline[1]['img_scale'] = (512, 512)

cfg.data.samples_per_gpu = 16

cfg.optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
cfg.lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 10,
    periods=[20,12,12,12,12],
    restart_weights = [1,0.7,0.6,0.5,0.4],
    min_lr_ratio=2e-6)

cfg.seed= 42
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/efficient_b0_BiFPN'

cfg.model.bbox_head.num_classes = 11

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)

#group_name = '아무거나'; ##그룹이름(run 여러개를 그룹으로 묶어줄 수 있습니다)
project_name = 'stage_3_object_det'; ##프로젝트 이름
run_name = 'efficient_b0_BiFPN'; ##run 이름stage_3_object_det
config_list = {
    'epoch' : cfg.runner.max_epochs,
    'batch_size' :  cfg.data.samples_per_gpu,
    'optimizer' : cfg.optimizer,
    'optimizer_config' : cfg.optimizer_config,
    'lr_config' : cfg.lr_config
}
## 원하는 wandb.init() argument값 추가해주세요
#cfg.log_config.hooks[1].init_kwargs['group']=group_name   # group name(option)
cfg.log_config.hooks[1].init_kwargs['project'] = project_name
cfg.log_config.hooks[1].init_kwargs['name'] = run_name    # run name
cfg.log_config.hooks[1].init_kwargs['config'] = config_list    # config

model = build_detector(cfg.model)

datasets = [build_dataset(cfg.data.train)]

train_detector(model, datasets[0], cfg, distributed=False, validate=True)