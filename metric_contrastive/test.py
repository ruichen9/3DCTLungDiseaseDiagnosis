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


PATH = './checkpoints/last.ckpt'
my_ptl= my_ptlframe()
model = my_ptl.load_from_checkpoint(PATH)
model.eval()

#加载数据集
_, _, testloader = data_get('../Data', batch_size=1)


test_count = 0
correct = 0
for batch_idx, (imgs, labels) in enumerate(testloader):
    outputs = model(imgs)
    test_count += imgs.size(0)
    _, predicted = torch.max(outputs.data, dim=1)
    if labels == predicted:
        correct += 1

accuracy = correct / test_count
print(accuracy)
