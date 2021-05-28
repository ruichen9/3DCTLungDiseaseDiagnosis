# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 21:14:51 2020

@author: 12057
"""
import numpy as np
import torch

#mix-up 常规 ont-hot label数据
def mix_up_onehot_data(x, y, alpha = 1):
    """
    对于one-hot label数据mix-up x和y
    :param: x 自变量
    :param: y 因变量
    :param: alpha Beta分布系数
    return mixed_x mix-up后x
    return mixed_y mix-up后y
    """
    
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha) #混合系数服从Beta(alpha, alpha)分布
    else:
        lam = 1.

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)  #随机排序shuffle

    mixed_x = lam * x + (1 - lam) * x[index,:] #x对应mix-up
    mixed_y = lam * y + (1 - lam) * y[index,:] #y对应mix-up
    
    return mixed_x, mixed_y
    

#mix-up 非ont-hot label数据
def mix_up_data(x, y, alpha = 1):
    """
    对于非one-hot label数据mix-up x 
    需要mix-up 损失函数同时使用
    :param: x 自变量
    :param: y 因变量
    :param: alpha Beta分布系数
    return mixed_x mix-up后x
    return y_a, y_b mix-up对应的两个y
    return lam mix-up系数lambda
    """
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha) #混合系数服从Beta(alpha, alpha)分布
    else:
        lam = 1.
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size) #随机排序shuffle

    mixed_x = lam * x + (1 - lam) * x[index,:] #x对应mix-up
    y_a, y_b = y, y[index] #匹配的一对y

    return mixed_x, y_a, y_b, lam  


