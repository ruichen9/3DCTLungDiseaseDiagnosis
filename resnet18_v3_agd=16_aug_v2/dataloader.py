# -*- coding: utf-8 -*-
"""dataload.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BAa3AyfVzmR95havf_irbyNfAwt-U5fd
"""



import os
import numpy as np 
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from dataAugmentation import *

def glob_allfile(file_holder, file_type):
    all_file = []
    for root, dirs, files in os.walk(file_holder):
        for file in files:
            if os.path.splitext(file)[1] == file_type :
                all_file.append(os.path.join(root, file))
    return all_file

def Split_train_val_test(img_list, img_index, trainRatio = 0.8, randomSeed = 666):
    img_list = np.array(img_list)
    img_index = np.array(img_index)    
    img_num = len(img_list)

    train_num = int(img_num * trainRatio)
    index_ = list(range(img_num))
    np.random.seed(randomSeed)
    np.random.shuffle(index_)

    train_idx = index_[: train_num]
    val_idx = index_[train_num:]
    test_idx = index_
    
    img_train = img_list[train_idx]
    img_train_index = img_index[train_idx]
    img_val = img_list[val_idx]
    img_val_index = img_index[val_idx]
    img_test = img_list[test_idx]
    img_test_index = img_index[test_idx]
    return img_train, img_train_index, img_val, img_val_index, img_test, img_test_index

def get_image_index(file_holder = '/content/drive/MyDrive/pytorch-lightning/data/', file_type = 'npy', randomSeed = 111, trainRatio = 0.8):
    """
    获得文件路径下的图片名和对应标签,并划分测试和训练集
    """
    dic_class = {'xinguan':0, 'bingdu':1, 'jiehe':2, 'xijun':3, 'zhengchang':4}
    if file_type == 'pkl':
        image_list = glob_allfile(file_holder,'.pkl')
    else:    
        image_list = glob_allfile(file_holder,'.npy')
    image_index=[]
    for i,img in enumerate(image_list):
        if 'ISCOVID' in str.split(img,'-') or 'ISCOVID1001_1013' in str.split(img,'-') or 'ISCOVID1014_1025' in str.split(img,'-'):
            image_index.append(0)
        elif 'BINGDU' in str.split(img,'-'):
            image_index.append(1)
        elif 'JIEHE' in str.split(img,'-'):
            image_index.append(2)
        elif 'XIJUN' in str.split(img,'-'):
            image_index.append(3)
        elif 'YINXING' in  str.split(img,'-'):
            image_index.append(4)
        else:
            print('#',i)
            assert False,'unknown label'
    img_train, img_train_index, img_val, img_val_index, img_test, img_test_index = Split_train_val_test(image_list, image_index)
    # arrayindex_train = list(range(len(img_train)))
    # np.random.seed(randomSeed)
    # np.random.shuffle(arrayindex_train)
    # img_train = img_train[arrayindex_train]
    # img_train = img_train[arrayindex_train]
    return img_train, img_train_index, img_val, img_val_index, img_test, img_test_index, dic_class

def train_loader(path, pic_size = (96,128, 128), file_type = 'npy'):
    """
    生成loader
    """
    preprocess = transforms.Compose([
                     transforms.Lambda(lambda img: Padding3d(img)),
                     #transforms.Lambda(lambda img: rotate_3d(img)),
                     transforms.Lambda(lambda img: cut_3d(img)),
                     #transforms.Lambda(lambda img: flip_3d(img)),
                     #transforms.Lambda(lambda img: add_black_3d(img)),
                     transforms.Lambda(lambda img: normalize(img)),
                     transforms.Lambda(lambda img: zero_center(img)),
                     #transforms.Lambda(lambda img: tobgr0_255(img, -700, 1500)),
                     transforms.Lambda(lambda img: resize_size(img, pic_size)),
                     transforms.Lambda(lambda img: tohwd(img)),
                     transforms.ToTensor(),
                 ])
    if file_type == 'pkl':
        with open(path, 'rb') as pkl_file:        
            img_array = pickle.load(pkl_file)
    else:
        img_array = np.load(path)
    img_tensor = preprocess(img_array)
    img_tensor = torch.tensor(img_tensor, dtype = torch.float32)
    D, H, W = list(img_tensor.shape)
    img_tensor = img_tensor.view(1, D, H, W)
    return img_tensor


def val_loader(path, pic_size = (96,128, 128), file_type = 'npy'):
    """
    生成loader
    """
    preprocess = transforms.Compose([
                     transforms.Lambda(lambda img: Padding3d(img)),
                     #transforms.Lambda(lambda img: rotate_3d(img)),
                     #transforms.Lambda(lambda img: cut_3d(img)),
                     #transforms.Lambda(lambda img: flip_3d(img)),
                     #transforms.Lambda(lambda img: add_black_3d(img)),
                     transforms.Lambda(lambda img: normalize(img)),
                     transforms.Lambda(lambda img: zero_center(img)),
                     #transforms.Lambda(lambda img: tobgr0_255(img, -700, 1500)),
                     transforms.Lambda(lambda img: resize_size(img, pic_size)),
                     transforms.Lambda(lambda img: tohwd(img)),
                     transforms.ToTensor(),
                 ])
    if file_type == 'pkl':
        with open(path, 'rb') as pkl_file:        
            img_array = pickle.load(pkl_file)
    else:
        img_array = np.load(path)
    img_tensor = preprocess(img_array)
    img_tensor = torch.tensor(img_tensor, dtype = torch.float32)
    D, H, W = list(img_tensor.shape)
    img_tensor = img_tensor.view(1, D, H, W)
    return img_tensor


def test_loader(path, pic_size = (96,128, 128), file_type = 'npy'):
    """
    生成loader
    """
    preprocess = transforms.Compose([
                     transforms.Lambda(lambda img: Padding3d(img)),
                     #transforms.Lambda(lambda img: rotate_3d(img)),
                     #transforms.Lambda(lambda img: cut_3d(img)),
                     #transforms.Lambda(lambda img: flip_3d(img)),
                     #transforms.Lambda(lambda img: add_black_3d(img)),
                     transforms.Lambda(lambda img: normalize(img)),
                     transforms.Lambda(lambda img: zero_center(img)),
                     #transforms.Lambda(lambda img: tobgr0_255(img, -700, 1500)),
                     transforms.Lambda(lambda img: resize_size(img, pic_size)),
                     transforms.Lambda(lambda img: tohwd(img)),
                     transforms.ToTensor(),
                 ])
    if file_type == 'pkl':
        with open(path, 'rb') as pkl_file:        
            img_array = pickle.load(pkl_file)
    else:
        img_array = np.load(path)
    img_tensor = preprocess(img_array)
    img_tensor = torch.tensor(img_tensor, dtype = torch.float32)
    D, H, W = list(img_tensor.shape)
    img_tensor = img_tensor.view(1, D, H, W)
    return img_tensor

class trainset(Dataset):
    def __init__(self, img_train, img_train_index, loader = train_loader):
        #定义好 image 的路径
        self.images = img_train
        self.target = img_train_index
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)


class valset(Dataset):
    def __init__(self, img_val, img_val_index, loader = val_loader):
        #定义好 image 的路径
        self.images = img_val
        self.target = img_val_index
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)


class testset(Dataset):
    def __init__(self, img_test, img_test_index, loader = test_loader):
        #定义好 image 的路径
        self.images = img_test
        self.target = img_test_index
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)

def data_get(root_folder , batch_size):
    """
    获取数据
    """
    img_train, img_train_index, img_val, img_val_index, img_test, img_test_index, dic_class = get_image_index(file_holder = root_folder, file_type ='npy')
    train_data = trainset(img_train, img_train_index)
    val_data = valset(img_val, img_val_index)
    test_data = testset(img_test, img_test_index)
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    return trainloader, valloader, testloader