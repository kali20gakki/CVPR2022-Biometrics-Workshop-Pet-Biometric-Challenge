import os
import random
from random import shuffle
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import json
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.utils.data.sampler import Sampler
import torch
from collections import defaultdict
import albumentations as A

def load_train_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    ids = [i for i in data]
    return data, ids

def load_train_data2(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    data_source = []
    for dog_id in data:
        for da in data[dog_id]:
            data_source.append([da, dog_id])

    return data_source


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

train_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

test_transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

# albu_transform = A.Compose([
#     A.OneOf([
#         A.HorizontalFlip(p=0.8),
#         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.8),
#         A.RandomBrightnessContrast(p=0.8),
#         A.IAASharpen(p=0.8),
#         A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, always_apply=False, p=0.8),
#         A.Blur(blur_limit=5, always_apply=False, p=0.8),
#     ], p = 1.0)
# ])




albu_transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.OneOf([ # 色彩
                            A.ToGray(p=0.1),
                            A.ChannelShuffle(p=0.5),
                            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
                            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                            A.RandomBrightnessContrast(p=0.5),
                            A.FancyPCA(alpha=0.1, always_apply=False, p=0.5),
                        ], p = 0.5),

                        A.OneOf([ # 翻转
                            A.Rotate(limit=45, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
                            A.Cutout(num_holes=3, max_h_size=32, max_w_size=32, fill_value=114, p=0.5),
                            A.PiecewiseAffine(p=0.5),
                            A.Perspective(p=0.5),
                            A.Affine(p=0.5),
                            A.RandomRotate90(p=0.5),
                            A.RandomResizedCrop(224, 224, scale=(0.4, 0.9), ratio=(0.75, 1.3333333333), interpolation=1, always_apply=False, p=0.5),
                        ], p = 0.5),

                        A.OneOf([ # 质量
                            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
                            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, elementwise=False, always_apply=False, p=0.5),
                            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
                            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5),
                            A.ImageCompression(quality_lower=80, quality_upper=90, p=0.5),
                            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5),
                            A.Posterize(num_bits=4, always_apply=False, p=0.5),

                            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, always_apply=False, p=0.5),
                            A.MotionBlur(p=0.5),
                            A.MedianBlur(p=0.5),
                            A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, always_apply=False, mode='fast', p=0.5),
                            A.Blur(blur_limit=7, always_apply=False, p=0.5),
                        ], p = 0.5),
                    ])


pseudo_albu_transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.OneOf([ # 色彩
                            A.ToGray(p=0.1),
                            A.ChannelShuffle(p=0.5),
                            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
                            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                            A.RandomBrightnessContrast(p=0.5),
                            A.FancyPCA(alpha=0.1, always_apply=False, p=0.5),
                        ], p = 0.5),
                        A.OneOf([ # 翻转
                            A.Rotate(limit=45, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
                            A.Cutout(num_holes=3, max_h_size=32, max_w_size=32, fill_value=114, p=0.8),
                            A.PiecewiseAffine(p=0.5),
                            A.Perspective(p=0.5),
                            A.Affine(p=0.5),
                            A.RandomRotate90(p=0.5),
                            A.RandomResizedCrop(224, 224, scale=(0.4, 0.9), ratio=(0.75, 1.3333333333), interpolation=1, always_apply=False, p=0.5),
                        ], p = 0.5),
                    ])


############################# New Aug #######################################
# albu_transform = A.Compose([
#                         A.HorizontalFlip(p=0.5),
#                         A.OneOf([ # 色彩
#                             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
#                             A.RandomBrightnessContrast(p=0.5),
#                         ], p = 0.5),

#                         A.OneOf([ # 翻转
#                             #A.Rotate(limit=45, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
#                             A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
#                             A.Cutout(num_holes=3, max_h_size=32, max_w_size=32, fill_value=114, p=0.5),
#                             A.RandomResizedCrop(224, 224, scale=(0.4, 0.9), ratio=(0.75, 1.3333333333), interpolation=1, always_apply=False, p=0.5),
#                         ], p = 0.5),

#                         A.OneOf([ # 质量
#                             #A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, elementwise=False, always_apply=False, p=0.5),
#                             A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
#                             # A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5),
#                             A.ImageCompression(quality_lower=80, quality_upper=90, p=0.5),

#                             A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, always_apply=False, p=0.5),
#                             A.MotionBlur(p=0.5),
#                             A.MedianBlur(p=0.5),
#                             A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, always_apply=False, mode='fast', p=0.5),
#                             A.Blur(blur_limit=7, always_apply=False, p=0.5),
#                         ], p = 0.5),
#                     ])


# pseudo_albu_transform = A.Compose([
#                         A.HorizontalFlip(p=0.5),
#                         A.OneOf([ # 色彩
#                             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
#                             A.RandomBrightnessContrast(p=0.5),
#                         ], p = 0.5),
#                         A.OneOf([ # 翻转
#                             A.Rotate(limit=45, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
#                             A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
#                             A.Cutout(num_holes=3, max_h_size=32, max_w_size=32, fill_value=114, p=0.8),
#                             A.PiecewiseAffine(p=0.5),
#                             A.Perspective(p=0.5),
#                             A.Affine(p=0.5),
#                             A.RandomRotate90(p=0.5),
#                             A.RandomResizedCrop(224, 224, scale=(0.4, 0.9), ratio=(0.75, 1.3333333333), interpolation=1, always_apply=False, p=0.5),
#                         ], p = 0.5),
#                     ])


class BiometricsDataset(Dataset):
    def __init__(self, train_json_data, ids, img_dir, imgsz, p = 0.5, is_aug = True):
        super(BiometricsDataset, self).__init__()
        self.train_json_data = train_json_data
        self.img_dir = img_dir
        self.imgsz = imgsz
        self.ids = ids
        self.p = p
        self.transforms = train_transforms
        self.is_aug = is_aug
        self.albu_transform = albu_transform

    def __getitem__(self, index):
        dog_id = self.ids[index]
        img_names = self.train_json_data[dog_id]

        if random.random() >= self.p: # 匹配pair
            imgA_name = img_names[0]
            imgB_name = img_names[1]
            match = 1
        else:
            imgA_name = random.choice(img_names)
            index_ = random.randint(0, index - 1) if index > len(self.train_json_data) / 2 else random.randint(index + 1, len(self.train_json_data) - 1)
            img_names = self.train_json_data[self.ids[index_]]
            imgB_name = random.choice(img_names)
            match = 0

        imgA_path = os.path.join(self.img_dir, imgA_name)
        imgB_path = os.path.join(self.img_dir, imgB_name)

        imgA = np.array(Image.open(imgA_path).convert('RGB'))
        imgB = np.array(Image.open(imgB_path).convert('RGB'))

        if self.is_aug and match:
            imgA = self.albu_transform(image=imgA)["image"]
            imgB = self.albu_transform(image=imgB)["image"]

        # resize to same size
        imgA = letterbox(imgA, new_shape=self.imgsz)[0]
        imgB = letterbox(imgB, new_shape=self.imgsz)[0]
        
        imgA = self.transforms(imgA)
        imgB = self.transforms(imgB)

        return imgA, imgB, torch.tensor(np.array(match))


    def __len__(self):
        return len(self.train_json_data)


class BiometricsClsDataset(Dataset):
    def __init__(self, train_json_data, ids, img_dir, imgsz, p = 0.5, is_aug = True):
        super(BiometricsClsDataset, self).__init__()
        self.train_json_data = train_json_data
        self.img_dir = img_dir
        self.imgsz = imgsz
        self.ids = ids
        self.p = p
        self.transforms = train_transforms
        self.is_aug = is_aug
        self.albu_transform = albu_transform

        self.start_id = int(min(train_json_data.keys()))
        print(f"Dog ID SUM = {len(train_json_data)}")

    def __getitem__(self, index):
        dog_id = self.ids[index]
        img_names = self.train_json_data[dog_id]
        imgA_cls = int(dog_id) - self.start_id # id < 1000 验证集

        if random.random() >= self.p: # 匹配pair
            imgA_name, imgB_name = random.sample(img_names, k=2)
            imgB_cls = imgA_cls
            match = 1
        else:
            imgA_name = random.choice(img_names)
            index_ = random.randint(0, index - 1) if index > len(self.train_json_data) / 2 else random.randint(index + 1, len(self.train_json_data) - 1)
            dog_id = self.ids[index_]
            imgB_cls = int(dog_id) - self.start_id # id < 1000 验证集
            img_names = self.train_json_data[dog_id]
            imgB_name = random.choice(img_names)
            match = 0

        imgA_path = os.path.join(self.img_dir, imgA_name)
        imgB_path = os.path.join(self.img_dir, imgB_name)

        imgA = np.array(Image.open(imgA_path).convert('RGB'))
        imgB = np.array(Image.open(imgB_path).convert('RGB'))

        # resize to same size
        imgA = letterbox(imgA, new_shape=self.imgsz)[0]
        imgB = letterbox(imgB, new_shape=self.imgsz)[0]

        if self.is_aug and match:
            imgA = self.albu_transform(image=imgA)["image"]
            imgB = self.albu_transform(image=imgB)["image"]

        imgA = self.transforms(imgA)
        imgB = self.transforms(imgB)
        
        # imgA_cls_onehot = F.one_hot(torch.tensor(np.array(imgA_cls)), num_classes=5000)
        # imgB_cls_onehot = F.one_hot(torch.tensor(np.array(imgB_cls)), num_classes=5000)
        
        return imgA, imgB, torch.tensor(np.array(match)).long(), torch.tensor(np.array(imgA_cls)).long(), torch.tensor(np.array(imgB_cls)).long()


    def __len__(self):
        return len(self.train_json_data)


class BiometricsValDataset(Dataset):
    def __init__(self, val_path, img_dir, imgsz=640):
        super(BiometricsValDataset, self).__init__()
        self.img_dir = img_dir
        self.imgsz = imgsz
        self.transforms = train_transforms
        with open(val_path, 'r') as f:
            self.val_data = json.load(f)

    def __getitem__(self, index):
        (imgA_name, imgB_name), label = self.val_data[index]['pair'], self.val_data[index]['label']
        imgA_path = os.path.join(self.img_dir, imgA_name)
        imgB_path = os.path.join(self.img_dir, imgB_name)

        imgA = np.array(Image.open(imgA_path).convert('RGB'))
        imgB = np.array(Image.open(imgB_path).convert('RGB'))

        # resize to same size
        imgA = letterbox(imgA, new_shape=self.imgsz)[0]
        imgB = letterbox(imgB, new_shape=self.imgsz)[0]

        imgA = self.transforms(imgA)
        imgB = self.transforms(imgB)

        return imgA, imgB, torch.tensor(np.array(label))

    def __len__(self):
        return len(self.val_data)



class BiometricsTestDataset(Dataset):
    def __init__(self, test_path, img_dir, imgsz=640):
        super(BiometricsTestDataset, self).__init__()
        self.img_dir = img_dir
        self.imgsz = imgsz
        self.transforms = train_transforms
        with open(test_path, 'r') as f:
            self.val_data = json.load(f)

    def __getitem__(self, index):
        imgA_name, imgB_name = self.val_data[index]['pair']

        imgA_path = os.path.join(self.img_dir, imgA_name)
        imgB_path = os.path.join(self.img_dir, imgB_name)

        imgA = np.array(Image.open(imgA_path).convert('RGB'))
        imgB = np.array(Image.open(imgB_path).convert('RGB'))

        # resize to same size
        imgA = letterbox(imgA, new_shape=self.imgsz)[0]
        imgB = letterbox(imgB, new_shape=self.imgsz)[0]

        imgA = self.transforms(imgA)
        imgB = self.transforms(imgB)

        return imgA, imgB, imgA_name, imgB_name

    def __len__(self):
        return len(self.val_data)


def random_w_cut(img, value = 114):
    img = img.copy()
    
    if random.random() > 0.5:
        H, W, _ = img.shape
        ratio = random.uniform(0.1, 0.3)
        img[:,:int(ratio * W),:] = value
        img[:,int(W - ratio * W):,:] = value

    return img



class BiometricsClsDataset2(Dataset):
    def __init__(self, data_source, img_dir, imgsz, p = 0.5, is_aug = True, use_w_cut = False):
        super(BiometricsClsDataset2, self).__init__()
        if use_w_cut:
            print('use random_w_cut')
        self.use_w_cut = use_w_cut
        self.data_source = data_source
        self.img_dir = img_dir
        self.imgsz = imgsz
        self.p = p
        self.transforms = train_transforms
        self.is_aug = is_aug
        self.albu_transform = albu_transform


    def __getitem__(self, index):
        img_name, dog_id = self.data_source[index]

        img_path = os.path.join(self.img_dir, img_name)

        img = np.array(Image.open(img_path).convert('RGB'))

        if self.is_aug and int(dog_id) <= 5999:
            img = self.albu_transform(image=img)["image"]
            if self.use_w_cut:
                img = random_w_cut(img)
        elif self.is_aug and int(dog_id) > 5999:
            img = pseudo_albu_transform(image=img)["image"]

        # resize to same size
        img = letterbox(img, new_shape=self.imgsz)[0]

        img = self.transforms(img)

        return img, torch.tensor(np.array(int(dog_id))).long()


    def __len__(self):
        return len(self.data_source)


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.
    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances


if __name__ == '__main__':
    data_source = load_train_data2('dataset/pet_biometric_challenge_2022/train/train_data_pseudo_v5.json')
    print(len(data_source))
    # train_datset = BiometricsClsDataset2(data_source, 'dataset/pet_biometric_challenge_2022/train/images', 224, p = 0.5, is_aug=True)

    # train_loader = Data.DataLoader(
    #     dataset=train_datset,  
    #     batch_size=12,       
    #     shuffle=False,     
    #     num_workers=3,
    #     drop_last=False,
    #     sampler=RandomIdentitySampler(data_source, 6)
    # )

    # print(len(train_loader))
    # vis = []
    # for i, (img, label) in enumerate(train_loader):
    #     pass
