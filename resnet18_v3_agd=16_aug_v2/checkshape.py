
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

def Split_train_test(img_list, img_index, trainRatio = 0.8, randomSeed = 666):
    img_list = np.array(img_list)
    img_index = np.array(img_index)    
    img_num = len(img_list)

    train_num = int(img_num * trainRatio)
    index_ = list(range(img_num))
    np.random.seed(randomSeed)
    np.random.shuffle(index_)

    train_idx = index_[: train_num]
    test_idx = index_[train_num: ]
    
    img_train = img_list[train_idx]
    img_train_index = img_index[train_idx]
    img_test = img_list[test_idx]
    img_test_index = img_index[test_idx]
    return img_train, img_train_index, img_test, img_test_index

def get_image_index(file_holder = '../Data', file_type = 'npy', randomSeed = 111, trainRatio = 0.8):
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
    img_train,img_train_index,img_test,img_test_index = Split_train_test(image_list,image_index)
    # arrayindex_train = list(range(len(img_train)))
    # np.random.seed(randomSeed)
    # np.random.shuffle(arrayindex_train)
    # img_train = img_train[arrayindex_train]
    # img_train = img_train[arrayindex_train]
    return img_train, img_train_index, img_test, img_test_index, dic_class

def train_loader(path, pic_size = (96,128, 128), file_type = 'npy'):
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

img_train, img_train_index, img_test, img_test_index, dic_class = get_image_index()


img_array = np.load(img_train[0])
preprocess = transforms.Compose([
                    transforms.Lambda(lambda img: Padding3d(img)),
                    #transforms.Lambda(lambda img: rotate_3d(img)),
                    #transforms.Lambda(lambda img: cut_3d(img)),
                    #transforms.Lambda(lambda img: flip_3d(img)),
                    #transforms.Lambda(lambda img: add_black_3d(img)),
                    transforms.Lambda(lambda img: normalize(img)),
                    transforms.Lambda(lambda img: zero_center(img)),
                    #transforms.Lambda(lambda img: tobgr0_255(img, -700, 1500)),
                    transforms.Lambda(lambda img: resize_size(img, (96,128, 128))),
                    transforms.Lambda(lambda img: tohwd(img)),
                    transforms.ToTensor(),
                ])
img_tensor = preprocess(img_array)
img_tensor = torch.tensor(img_tensor, dtype=torch.float32)
D, H, W = list(img_tensor.shape)
img_tensor = img_tensor.view(1, D, H, W)
print(img_tensor.shape)

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        


    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        # x = self.bn1(x)
        # x = self.relu(x)
        # if not self.no_max_pool:
        #     x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)

        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

aaa = torch.randn(8, 16, 3, 112, 112)
model = ResNet(block=BasicBlock,
               layers=[2,2,2,2],
               block_inplanes=get_inplanes(),
               n_input_channels=1,
               conv1_t_size=7,
               conv1_t_stride=1,
               no_max_pool=False,
               shortcut_type='B',
               widen_factor=1.0,
               n_classes=5)
model(aaa)