import torch
import torch.nn as nn
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


def testMetric(model, testloader,Criterion,nepoch=5):
    # default model is in the cuda
    # test loss on the test set
    test_loss_i = 0
    model.eval()
    test_count = 0
    for epoch in range(nepoch):
        for batch_idx, (imgs1,imgs2, targets) in enumerate(testloader):
            inputs1,inputs2, targets = imgs1.to("cuda"),imgs2.to("cuda"), targets.to("cuda")
            Enbed1 = model(inputs1) #前向传播
            Enbed2 = model(inputs2)
            lossOntest = Criterion(Enbed1,Enbed2,targets)
            test_loss_i += lossOntest.data.item()
            test_count += inputs1.size(0)
    return test_loss_i/(nepoch*test_count)



def testClassifier(model, testloader,Criterion):
    # default model is in the cuda
    # test loss on the test set
    test_loss_i = 0
    model.eval()
    test_count = 0
    accuracy=[]
    for batch_idx, (imgs1, label) in enumerate(testloader):
        inputs, targets = imgs1.to("cuda"), label.to("cuda")
        outputs = model(inputs)
        lossOntest = Criterion(outputs,targets)
        test_loss_i += lossOntest.data.item()
        test_count += inputs1.size(0)
        _, predicted = torch.max(outputs.data, dim = 1) #获得预测结果，结果为批次数据, 所以行最大(一行一个结果)
        correct += predicted.eq(targets.data).c���N^���`���s�_��!�6>�W}�v��_��!�6>�W}�v�Z@���;B3Fw5��V��item()获取值
    return test_loss_i/(test_count), correct/test_count
    