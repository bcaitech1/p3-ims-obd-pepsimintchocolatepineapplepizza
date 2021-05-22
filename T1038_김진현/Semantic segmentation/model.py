import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import label_accuracy_score
import cv2

from tqdm import tqdm

import gc

import math
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np
import pandas as pd

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import segmentation_models_pytorch as smp

import wandb

class CosineAnnealingWarmUpRestarts(_LRScheduler):
        def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
            if T_0 <= 0 or not isinstance(T_0, int):
                raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
            if T_mult < 1 or not isinstance(T_mult, int):
                raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
            if T_up < 0 or not isinstance(T_up, int):
                raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
            self.T_0 = T_0
            self.T_mult = T_mult
            self.base_eta_max = eta_max
            self.eta_max = eta_max
            self.T_up = T_up
            self.T_i = T_0
            self.gamma = gamma
            self.cycle = 0
            self.last_epoch = last_epoch
            self.T_cur = last_epoch
            super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
            
        
        def get_lr(self):
            if self.T_cur == -1:
                return self.base_lrs
            elif self.T_cur < self.T_up:
                return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
            else:
                return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                        for base_lr in self.base_lrs]

        def step(self, epoch=None):
            if epoch is None:
                epoch = self.last_epoch + 1
                self.T_cur = self.T_cur + 1
                if self.T_cur >= self.T_i:
                    self.cycle += 1
                    self.T_cur = self.T_cur - self.T_i
                    self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
            else:
                if epoch >= self.T_0:
                    if self.T_mult == 1:
                        self.T_cur = epoch % self.T_0
                        self.cycle = epoch // self.T_0
                    else:
                        n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                        self.cycle = n
                        self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                        self.T_i = self.T_0 * self.T_mult ** (n)
                else:
                    self.T_i = self.T_0
                    self.T_cur = epoch
                    
            self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
            self.last_epoch = math.floor(epoch)
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

wandb.init(project='seg', entity='jihopark')
config = wandb.config
encoder_name = 'timm-efficientnet-b4'
decoder_name = 'FPN'

plt.rcParams['axes.grid'] = False

print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())

device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 사용 가능 여부에 따라 device 정보 저장

batch_size = 8   # Mini-batch size
num_epochs = 60
learning_rate = 0
num_classes = 12

config.learning_rate = 0

# seed 고정
random_seed = 21
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

dataset_path = '../input/data'
anns_file_path = dataset_path + '/' + 'train.json'

# Read annotations
with open(anns_file_path, 'r') as f:
    dataset = json.loads(f.read())

categories = dataset['categories']
anns = dataset['annotations']
imgs = dataset['images']
nr_cats = len(categories)
nr_annotations = len(anns)
nr_images = len(imgs)

# Load categories and super categories
cat_names = []
super_cat_names = []
super_cat_ids = {}
super_cat_last_name = ''
nr_super_cats = 0
for cat_it in categories:
    cat_names.append(cat_it['name'])
    super_cat_name = cat_it['supercategory']
    # Adding new supercat
    if super_cat_name != super_cat_last_name:
        super_cat_names.append(super_cat_name)
        super_cat_ids[super_cat_name] = nr_super_cats
        super_cat_last_name = super_cat_name
        nr_super_cats += 1

print('Number of super categories:', nr_super_cats)
print('Number of categories:', nr_cats)
print('Number of annotations:', nr_annotations)
print('Number of images:', nr_images)

# Count annotations
cat_histogram = np.zeros(nr_cats,dtype=int)
for ann in anns:
    cat_histogram[ann['category_id']] += 1

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(5,5))

# Convert to DataFrame
df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
df = df.sort_values('Number of annotations', 0, False)

# Plot the histogram
plt.title("category distribution of train set ")
plot_1 = sns.barplot(x="Number of annotations", y="Categories", data=df, label="Total", color="b")

# category labeling 
sorted_temp_df = df.sort_index()

# background = 0 에 해당되는 label 추가 후 기존들을 모두 label + 1 로 설정
sorted_df = pd.DataFrame(["Backgroud"], columns = ["Categories"])
sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)


# class (Categories) 에 따른 index 확인 (0~11 : 총 12개)
sorted_df

category_names = list(sorted_df.Categories)

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)# .astype(np.float32)
        # images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)
            # print("image_infos['id'] : {}".format(image_infos['id']) )
            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
            masks = masks.astype(np.float32)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            images = images/255.0
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            images = images/255.0
            return images, image_infos
    
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

# train.json / validation.json / test.json 디렉토리 설정
train_paths = [dataset_path + f'/train_data{i}.json' for i in range(5)]
val_paths = [dataset_path + f'/valid_data{i}.json' for i in range(5)]
test_path = dataset_path + '/test.json'

test_transform = A.Compose([
                            ToTensorV2()
                            ])

# test dataset
test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        num_workers=4,
                                        collate_fn=collate_fn)

for fold, (train_path, val_path) in enumerate(zip(train_paths, val_paths)):
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'

    train_transform = A.Compose([
                             A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p = 0.5),
                             A.VerticalFlip(p=0.1),
                             A.HorizontalFlip(p=0.5),
							 A.RandomBrightness(),
                             A.ShiftScaleRotate(shift_limit=0, scale_limit=(-1e-6, 0.1),rotate_limit=30),
                             A.GridDistortion(border_mode=cv2.BORDER_CONSTANT),
                             ToTensorV2()
							])

    val_transform = A.Compose([
                            ToTensorV2()
                            ])

    # train dataset
    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)

    # validation dataset
    val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    # model 불러오기
    # 출력 레이블 수 정의 (classes = 12)
    model = smp.FPN(encoder_name=encoder_name, classes=12, encoder_weights="noisy-student", activation=None)
    model = model.to(device)
    wandb.watch(model, log_freq=100)
    def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, scheduler, saved_dir, val_every, device):
        print(f'Start training..fold{fold+1}')
        best_mIoU = 0
        best_epoch = 0
        for epoch in tqdm(range(num_epochs)):
            model.train()
            for step, (images, masks, _) in enumerate(data_loader):
                images = torch.stack(images)       # (batch, channel, height, width)
                masks = torch.stack(masks).long()  # (batch, channel, height, width)
                
                # gpu 연산을 위해 device 할당
                images, masks = images.to(device), masks.to(device)
                    
                # inference
                outputs = model(images)
                
                # loss 계산 (cross entropy loss)
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # step 주기에 따른 loss 출력
                if (step + 1) % 25 == 0:
                    wandb.log({f"fold {fold} loss": loss.item()})
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                        epoch+1, num_epochs, step+1, len(train_loader), loss.item()))
            
            # validation 주기에 따른 loss 출력 및 best model 저장
            if (epoch + 1) % val_every == 0:
                mIoU = validation(epoch + 1, model, val_loader, criterion, device)
                wandb.log({f"fold {fold} mIoU": mIoU})
                if mIoU > best_mIoU:
                    best_epoch = epoch + 1
                    print('Best performance at epoch: {}'.format(best_epoch))
                    print('Save model in', saved_dir)
                    best_mIoU = mIoU
                    wandb.log({f"fold {fold} best mIoU": best_mIoU})
                    save_model(model, saved_dir)
                if epoch + 1 - best_epoch >= 15:
                    print("Early Stop!!!!")
                    return
            print(f'fold {fold} lr={optimizer.param_groups[0]["lr"]}')
            wandb.log({f'fold {fold} lr': optimizer.param_groups[0]['lr']})
            scheduler.step()


    def validation(epoch, model, data_loader, criterion, device):
        print('Start validation fold{} #{}'.format(fold+1, epoch))
        model.eval()
        with torch.no_grad():
            total_loss = 0
            cnt = 0
            mIoU_list = []
            for step, (images, masks, _) in enumerate(data_loader):
                
                images = torch.stack(images)       # (batch, channel, height, width)
                masks = torch.stack(masks).long()  # (batch, channel, height, width)

                images, masks = images.to(device), masks.to(device)            

                outputs = model(images)
                loss = criterion(outputs, masks)
                total_loss += loss
                cnt += 1
                
                outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()

                mIoUs = label_accuracy_score(masks.detach().cpu().numpy(), outputs, n_class=12)
                mIoU_list += mIoUs
            avrg_loss = total_loss / cnt
            print('Validation {} #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(fold+1, epoch, avrg_loss, np.mean(mIoU_list)))

        return np.mean(mIoU_list)

    # 모델 저장 함수 정의
    val_every = 1 

    saved_dir = './saved'
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)
        
    def save_model(model, saved_dir, file_name=f'fold-{fold}-{encoder_name}-{decoder_name}.pt'):
        check_point = {'net': model.state_dict()}
        output_path = os.path.join(saved_dir, file_name)
        torch.save(model.state_dict(), output_path)

    # Loss function 정의
    criterion = nn.CrossEntropyLoss()

    # Optimizer 정의
    optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=15, T_mult=1, eta_max=0.0005, T_up=2, gamma=0.3)
    #train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, val_every, device)
    del train_loader, val_loader, train_dataset, val_dataset, model
    gc.collect()
    break

def test(model, data_loader, device, fold, outs=None):
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    model.eval()
    if fold >= 0:
        print(f'Start stack. fold{fold}')
    else:
        print('Start prediction')
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        if fold is 0:
            outs = [model(torch.stack(imgs).to(device)) for (imgs, _) in test_loader]
            return outs
        elif fold > 0:
            for step, (imgs, image_infos) in enumerate(test_loader):
                outs[step] += model(torch.stack(imgs).to(device))
            return outs
        else:
            for (out, (imgs, image_infos)) in zip(outs, test_loader):
                oms = torch.argmax(out, dim=1).detach().cpu().numpy()
                
                # resize (256 x 256)
                temp_mask = []
                for img, mask in zip(np.stack(imgs), oms):
                    transformed = transform(image=img, mask=mask)
                    mask = transformed['mask']
                    temp_mask.append(mask)

                oms = np.array(temp_mask)
                
                oms = oms.reshape([oms.shape[0], size*size]).astype(int)
                preds_array = np.vstack((preds_array, oms))
                
                file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array
model = smp.FPN(encoder_name=encoder_name, classes=12, encoder_weights="noisy-student", activation=None)
model = model.to(device)
model_paths = [f'./saved/fold-{fold}-{encoder_name}-{decoder_name}.pt' for fold in range(5)]
checkpoint = torch.load(model_paths[0], map_location=device)
model.load_state_dict(checkpoint)
result = test(model, test_loader, device, 0)

for fold in range(1, 5):
    checkpoint = torch.load(model_paths[fold], map_location=device)
    model.load_state_dict(checkpoint)
    result = test(model, test_loader, device, fold, result)

# test set에 대한 prediction
file_names, preds = test(model, test_loader, device, -1, result)
# sample_submisson.csv 열기
submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)
# PredictionString 대입
for file_name, string in zip(file_names, preds):
    submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                ignore_index=True)

# submission.csv로 저장
submission.to_csv(f"./submission/K-fold-{encoder_name}-{decoder_name}.csv", index=False)
