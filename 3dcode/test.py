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
            Enbed1 = model(inputs1) #å‰å‘ä¼ æ’­
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
        _, predicted = torch.max(outputs.data, dim = 1) #è·å¾—é¢„æµ‹ç»“æœï¼Œç»“æœä¸ºæ‰¹æ¬¡æ•°æ®, æ‰€ä»¥è¡Œæœ€å¤§(ä¸€è¡Œä¸€ä¸ªç»“æœ)
        correct += predicted.eq(targets.data).cŸ÷¶N^±æë`™£sò_­Ç!¢6>ÉW}év¹ò_­Ç!¢6>ÉW}év¹Z@ÊĞç;B3Fw5†ÖV¼äitem()è·å–å€¼
    return test_loss_i/(test_count), correct/test_count
    