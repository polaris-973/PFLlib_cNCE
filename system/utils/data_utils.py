# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import ImageFilter
import random


# 定义 MultiViewGenerator 类
class MultiViewGenerator:
    """Create multiple crops of the same image"""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [x, self.base_transform(x)]
    
# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
        
# 定义类方法加载数据并应用数据增强
class DataHandler:
    def __init__(self, dataset, client_id, batch_size, n_views):
        self.dataset = dataset
        self.id = client_id
        self.batch_size = batch_size
        self.n_views = n_views

    def load_augmented_train_data(self):
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        # CIFAR-10
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])

        train_dataset = CustomDataset(train_data, transform=MultiViewGenerator(train_transform, self.n_views))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True, pin_memory=True)

        return train_loader

def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data

def read_client_data(dataset, idx, is_train=True):
    if "News" in dataset:
        return read_client_data_text(dataset, idx, is_train)
    elif "Shakespeare" in dataset:
        return read_client_data_Shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data
    

def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_Shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


# class GaussianBlur(object):
#     """blur a single image on CPU"""
#     def __init__(self, kernel_size):
#         radias = kernel_size // 2
#         kernel_size = radias * 2 + 1
#         self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
#                                 stride=1, padding=0, bias=False, groups=3)
#         self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
#                                 stride=1, padding=0, bias=False, groups=3)
#         self.k = kernel_size
#         self.r = radias

#         self.blur = nn.Sequential(
#             nn.ReflectionPad2d(radias),
#             self.blur_h,
#             self.blur_v
#         )

#         self.pil_to_tensor = transforms.ToTensor()
#         self.tensor_to_pil = transforms.ToPILImage()

#     def __call__(self, img):
#         img = self.pil_to_tensor(img).unsqueeze(0)

#         sigma = np.random.uniform(0.1, 2.0)
#         x = np.arange(-self.r, self.r + 1)
#         x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
#         x = x / x.sum()
#         x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

#         self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
#         self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

#         with torch.no_grad():
#             img = self.blur(img)
#             img = img.squeeze()

#         img = self.tensor_to_pil(img)

#         return img
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
def rand_bbox(size, lam):
    '''Getting the random box in CutMix'''
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_universum(images, labels, lamda=0.3, opt='mixup'):
    """Calculating Mixup-induced universum from a batch of images"""
    tmp = images.cpu()
    label = labels.cpu()
    bsz = tmp.shape[0]
    bs = len(label)
    class_images = [[] for i in range(max(label) + 1)]
    for i in label.unique():
        class_images[i] = np.where(label != i)[0]
    units = [tmp[random.choice(class_images[labels[i % bs]])] for i in range(bsz)]
    universum = torch.stack(units, dim=0).cuda()
    if not hasattr(opt, 'mix') or opt.mix == 'mixup':
        # Using Mixup
        universum = lamda * universum + (1 - lamda) * images
    else:
        # Using CutMix
        lam = 0
        while lam < 0.45 or lam > 0.55:
            # Since it is hard to control the value of lambda in CutMix,
            # we accept lambda in [0.45, 0.55].
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lamda)
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        universum[:, :, bbx1:bbx2, bby1:bby2] = images[:, :, bbx1:bbx2, bby1:bby2]
    return universum