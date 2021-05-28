import torch
import torch.nn as nn
from torch.utils import data
from torch import optim
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as ptl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import *
from models.resnet18 import *
from Resnet_lightning import *

ckpt_list = glob_allfile('./checkpoints', '.ckpt')
ckpt_list.sort()
#加载数据集
trainloader, valloader, testloader = data_get('../Data', batch_size=1)

accuracy_list = []

device = torch.device("cuda:2")
for i in range(len(ckpt_list)):
    count = 0
    correct = 0
    my_ptl = my_ptlframe()
    model = my_ptl.load_from_checkpoint(ckpt_list[i])
    model.to(device)
    model.eval()

    for batch_idx, (imgs, labels) in enumerate(testloader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        count += imgs.size(0)
        _, predicted = torch.max(outputs.data, dim=1)
        if labels == predicted:
            correct += 1
    accuracy = correct / count
    accuracy_list.append(accuracy)
    print(accuracy_list)

accuracy_list = np.array(accuracy_list)
np.save('accuracy_list.npy', accuracy_list)