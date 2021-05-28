import os
import numpy as np 
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from .dataAugmentation import *
import random


def glob_allfile(file_holder, file_type):
    all_file = []
    for root, dirs, files in os.walk(file_holder):
        for file in files:
            if os.path.splitext(file)[1] == file_type :
                all_file.append(os.path.join(root, file))
    return all_file


def Split_train_test(img_list, isclass, trainRatio = 0.8, randomSeed = 111):
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

def get_image_index(file_holder = '/home/sufedc_nvidia_newgyh/Public/npyBatch0609/', file_type = 'npy', randomSeed = 111, trainRatio = 0.8):
    """
    获得文件路径下的图片名和对应标签,并划分测试和训练集
    """
    dic_class = {'xinguan':0, 'bingdu':1, 'jiehe':2, 'xijun':3, 'zhengchang':4}
    if file_type == 'pkl':
        image_list0 = glob_allfile(os.path.join(file_holder, 'COVID'), '.pkl')
        image_list1 = glob_allfile(os.path.join(file_holder, 'BINGDU'), '.pkl')
        image_list2 = glob_allfile(os.path.join(file_holder, 'JIEHE'), '.pkl')
        image_list3 = glob_allfile(os.path.join(file_holder, 'XIJUN'), '.pkl')
        image_list4 = glob_allfile(os.path.join(file_holder, 'YINXING'), '.pkl')
    else:
        image_list0 = glob_allfile(os.path.join(file_holder, 'COVID'), '.npy')
        image_list1 = glob_allfile(os.path.join(file_holder, 'BINGDU'), '.npy')
        image_list2 = glob_allfile(os.path.join(file_holder, 'JIEHE'), '.npy')
        image_list3 = glob_allfile(os.path.join(file_holder, 'XIJUN'), '.npy')
        image_list4 = glob_allfile(os.path.join(file_holder, 'YINXING'), '.npy')      

    img_train0, img_train_index0, img_test0, img_test_index0 = Split_train_test(image_list0, isclass = 0 ,trainRatio = trainRatio, randomSeed = randomSeed)
    img_train1, img_train_index1, img_test1, img_test_index1 = Split_train_test(image_list1, isclass = 1 ,trainRatio = trainRatio, randomSeed = randomSeed)
    img_train2, img_train_index2, img_test2, img_test_index2 = Split_train_test(image_list2, isclass = 2 ,trainRatio = trainRatio, randomSeed = randomSeed)
    img_train3, img_train_index3, img_test3, img_test_index3 = Split_train_test(image_list3, isclass = 3 ,trainRatio = trainRatio, randomSeed = randomSeed)
    img_train4, img_train_index4, img_test4, img_test_index4 = Split_train_test(image_list4, isclass = 4 ,trainRatio = trainRatio, randomSeed = randomSeed)
    
    img_train = list(img_train0) + list(img_train1) + list(img_train2) + list(img_train3) + list(img_train4)
    img_train_index = img_train_index0 + img_train_index1 + img_train_index2 + img_train_index3 + img_train_index4
    img_test = list(img_test0) + list(img_test1) + list(img_test2) + list(img_test3) + list(img_test4)
    img_test_index = img_test_index0 + img_test_index1 + img_test_index2 + img_test_index3 + img_test_index4
    # arrayindex_train = list(range(len(img_train)))
    # np.random.seed(randomSeed)
    # np.random.shuffle(arrayindex_train)
    # img_train = img_train[arrayindex_train]
    # img_train = img_train[arrayindex_train]
    return img_train, img_train_index, img_test, img_test_index, dic_class



def train_loaderMetric(path, pic_size = (96,128, 128), file_type = 'npy'):
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


def test_loaderMetric(path, pic_size = (96,128, 128), file_type = 'npy'):
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

class trainsetMetric(Dataset):
    def __init__(self, img_train, img_train_index, loader = train_loaderMetric):
        #定义好 image 的路径
        self.images = img_train
        self.target = img_train_index
        self.loader = loader

    def __getitem__(self, index):
        ind1 = random.randint(0,len(self.images)-1)
        ind2 = random.randint(0,len(self.images)-1)
        fn1 = self.images[ind1]
        img1 = self.loader(fn1)
        target1 = self.target[ind1]
        fn2 = self.images[ind2]
        img2 = self.loader(fn2)
        target2 = self.target[ind2]
        target = 0
        if target1==target2:
            target = 1
        return img1,img2,target

    def __len__(self):
        return len(self.images)


class testsetMetric(Dataset):
    def __init__(self, img_test, img_test_index, loader = test_loaderMetric):
        #定义好 image 的路径
        self.images = img_test
        self.target = img_test_index
        self.loader = loader

    def __getitem__(self, index):
        ind1 = random.randint(0,len(self.images)-1)
        ind2 = random.randint(0,len(self.images)-1)
        fn1 = self.images[ind1]
        img1 = self.loader(fn1)
        target1 = self.target[ind1]
        fn2 = self.images[ind2]
        img2 = self.loader(fn2)
        target2 = self.target[ind2]
        target = 0
        if target1==target2:
            target = 1
        return img1,img2,target

    def __len__(self):
        return len(self.images)


def data_getMetric(root_folder , bbatch_size):
    """
    获取数据
    """
    img_train, img_train_index, img_test, img_test_index, dic_class = get_image_index(file_holder = root_folder, file_type ='npy')
    train_data = trainsetMetric(img_train, img_train_index)
    test_data = testsetMetric(img_test, img_test_index)
    trainloader = DataLoader(train_data, batch_size = bbatch_size, shuffle = True, num_workers = 0)
    testloader = DataLoader(test_data, batch_size = bbatch_size, shuffle = False, num_workers = 0)
    return trainloader, testloader
