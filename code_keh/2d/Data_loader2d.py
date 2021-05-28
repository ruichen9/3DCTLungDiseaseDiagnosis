# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:17:10 2020

@author: 12057
"""
import os
import numpy as np 
import pandas as pd
import pickle
import PIL

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch

from Data_Augment_array import *
from config import Config #获取配置


def glob_allfile(file_holder, file_type):
    """
    获取文件夹下包括子文件夹下的固定类型的所有文件
    :param: file_type 图片后缀名
    :param: file_holder 目标文件夹
    :return: all_file, 所有文件
    """
    all_file = []
    for root, dirs, files in os.walk(file_holder):
        for file in files:
            if os.path.splitext(file)[1] == file_type :
                all_file.append(os.path.join(root, file))
    return all_file

def Split_train_test(img_list, isclass = 1 ,trainRatio = 0.8, randomSeed = 111):
    
    img_index = [isclass] * len(img_list)
    img_list = np.array(img_list)
    img_index = np.array(img_index)
    
    img_num = len(img_list)

    train_num = int(img_num * trainRatio)
    
    index_ = list(range(img_num))
    np.random.seed(randomSeed + isclass)
    np.random.shuffle(index_)
    
    train_idx = index_[: train_num]
    test_idx = index_[train_num: ]
    
    img_train = img_list[train_idx]
    img_train_index = [isclass] * len(img_train)
    img_test = img_list[test_idx]
    img_test_index = [isclass] * len(img_test)
    
    return img_train, img_train_index, img_test, img_test_index

def get_image_index(file_holder = r'./ae_result', file_type = 'npy', trainRatio = 0.9 , randomSeed = 111):
    """
    获得文件路径下的图片名和对应标签,并划分测试和训练集
    """
    dic_class = {0:'xinguan', 1:'noxinguan'}
    if file_type == 'pkl':
        image_list0 = glob_allfile(os.path.join(file_holder, 'xinguan'), '.pkl')
    
        image_list1 = glob_allfile(os.path.join(file_holder, 'noxinguan'), '.pkl')
    else:
        image_list0 = glob_allfile(os.path.join(file_holder, 'xinguan'), '.npy')
    
        image_list1 = glob_allfile(os.path.join(file_holder, 'noxinguan'), '.npy')
    
    img_train0, img_train_index0, img_test0, img_test_index0 = Split_train_test(image_list0, isclass = 0 ,trainRatio = trainRatio, randomSeed = randomSeed)
    img_train1, img_train_index1, img_test1, img_test_index1 = Split_train_test(image_list1, isclass = 1 ,trainRatio = trainRatio, randomSeed = randomSeed)
    
    img_train = list(img_train0) + list(img_train1)
    img_train_index = img_train_index0 + img_train_index1
    img_test = list(img_test0) + list(img_test1)
    img_test_index = img_test_index0 + img_test_index1 
    # arrayindex_train = list(range(len(img_train)))
    # np.random.seed(randomSeed)
    # np.random.shuffle(arrayindex_train)
    # img_train = img_train[arrayindex_train]
    # img_train = img_train[arrayindex_train]
    
    test_dataframe = pd.DataFrame(columns = ['index', 'type', 'path'])
    for i in range(len(img_test)):
       test_dataframe.loc[len(test_dataframe)] = [i, dic_class[img_test_index[i]], img_test[i]]
    
    json_test = test_dataframe.to_json(orient = 'index')
    with open(r'./test_list_dae.json', 'w') as jsonFile:
         jsonFile.write(json_test)
                                           
    return img_train, img_train_index, img_test, img_test_index, dic_class

def train_loader(path, cfg, file_type = 'npy'):
    """
    生成loader
    """
    preprocess = transforms.Compose(
        [
        #传统数据增强
        #transforms.Lambda(lambda img: cut_random(img)), #随机裁剪 + 缩放
        #transforms.Lambda(lambda img: flip(img)), #随机翻转
        #transforms.Lambda(lambda img: add_black_frame(img)), #随机加黑边
        transforms.Lambda(lambda img: resize(img, resize_size = cfg.resize_size)), #调整大小一致，所有图片大小需要一致
        transforms.Lambda(lambda img: normalize(img)), #标准化：固定操作
        transforms.ToTensor()#tensor化：固定操作  函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
        ])

    if file_type == 'pkl':
        with open(path, 'rb') as pkl_file:        
            img_array = pickle.load(pkl_file)
    else:
        img_array = np.load(path)
    img_tensor = preprocess(img_array)
    img_tensor = torch.tensor(img_tensor, dtype = torch.float32)
    
    return img_tensor
  
def test_loader(path, cfg, file_type = 'npy'):
    """
    生成loader
    """
    preprocess = transforms.Compose([
        transforms.Lambda(lambda img: resize(img, resize_size = cfg.resize_size)), #调整大小一致，所有图片大小需要一致
        transforms.Lambda(lambda img: normalize(img)), #标准化：固定操作
        transforms.ToTensor() #tensor化：固定操作  函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
        ])

    if file_type == 'pkl':
        with open(path, 'rb') as pkl_file:        
            img_array = pickle.load(pkl_file)
    else:
        img_array = np.load(path)
        
    img_tensor = preprocess(img_array)
    img_tensor = torch.tensor(img_tensor, dtype = torch.float32)
    
    return img_tensor
    
class trainset(Dataset):
    def __init__(self, img_train, img_train_index, cfg, loader = train_loader):
        #定义好 image 的路径
        self.images = img_train
        self.target = img_train_index
        self.cfg = cfg
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn, cfg = self.cfg)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)


class testset(Dataset):
    def __init__(self, img_test, img_test_index, cfg, loader = test_loader):
        #定义好 image 的路径
        self.images = img_test
        self.target = img_test_index
        self.cfg = cfg
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn, cfg = self.cfg)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)

def data_get(config):
    """
    获取数据
    """
    img_train, img_train_index, img_test, img_test_index, dic_class = get_image_index(config.data_info, 'npy')
    train_data = trainset(img_train, img_train_index, config)
    
    """
    if config.distribution: #分布式训练
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        trainloader = DataLoader(train_data, batch_size = int(config.batch_size / cfg.distribution), shuffle = True, sampler = train_sampler)
    else:
        trainloader = DataLoader(train_data, batch_size = config.batch_size, shuffle = True)
    """
    trainloader = DataLoader(train_data, batch_size = config.batch_size, num_workers = 0, shuffle = True)
    test_data = testset(img_test, img_test_index, config)
    testloader = DataLoader(test_data, batch_size = 1, shuffle = False, num_workers = 0)
    
    return trainloader, testloader
    
    