# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
sys.path.append(r'/home/sufedc_nvidia_newgyh/apex')
from apex import amp
import torch
import torch.nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import PIL
import os
import time
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from config import Config #获取配置
#考虑到模型并行和预训练加载，所以模型的处理需要单独拿出来做我觉得是这样的
#model build

#model initweight
#model pretrain