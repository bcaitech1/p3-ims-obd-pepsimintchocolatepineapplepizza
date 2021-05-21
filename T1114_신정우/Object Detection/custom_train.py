from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)


classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
# config file 들고오기
# /opt/ml/code/mmdetection_trash/configs/trash/swin_t/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py
cfg = Config.fromfile('/opt/ml/code/mmdetection_trash/configs/trash/swin_t/swin_htc_Jinhyoen.py')

PREFIX = '../../input/data/'


project_name = 'obj' ##프로젝트 이름
run_name = 'swin_base_htc_00_jh'; ##run 이름
config_list = {
    'epoch' : cfg.runner.max_epochs,
    'batch_size' :  cfg.data.samples_per_gpu,
    'optimizer' : cfg.optimizer,
    'optimizer_config' : cfg.optimizer_config,
    'lr_config' : cfg.lr_config
}

cfg.log_config.hooks[1].init_kwargs['name'] = run_name    # run name
cfg.log_config.hooks[1].init_kwargs['config'] = config_list    # config

# dataset 바꾸기
cfg.data.train.classes = classes
cfg.data.train.seg_prefix = PREFIX
cfg.data.train.img_prefix = PREFIX
cfg.data.train.ann_file = PREFIX + 'train_data4.json'
# cfg.data.train.pipeline[2]['img_scale'] = (512, 512)

cfg.data.val.classes = classes
cfg.data.val.img_prefix = PREFIX
cfg.data.val.seg_prefix = PREFIX
cfg.data.val.ann_file = PREFIX + 'valid_data4.json'
# cfg.data.val.pipeline[1]['img_scale'] = (512, 512)

cfg.data.test.classes = classes
cfg.data.test.img_prefix = PREFIX
cfg.data.test.ann_file = PREFIX + 'test.json'
# cfg.data.test.pipeline[1]['img_scale'] = (512, 512)

cfg.data.samples_per_gpu = 2

#num classes
cfg.model.roi_head.bbox_head[0].num_classes = 11
cfg.model.roi_head.bbox_head[1].num_classes = 11
cfg.model.roi_head.bbox_head[2].num_classes = 11

cfg.seed=2020
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/swin_htc_0_jinhyoen'

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.model.pretrained = '/opt/ml/code/mmdetection_trash/work_dirs/swin_b_cascade_rcnn/swin_base_patch4_window7_224_22k.pth'

model = build_detector(cfg.model)

datasets = [build_dataset(cfg.data.train)]

train_detector(model, datasets[0], cfg, distributed=False, validate=True)