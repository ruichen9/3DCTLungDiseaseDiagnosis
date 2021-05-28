# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 22:41:37 2020

@author: 12057
"""
import cv2
import numpy as np
from skimage import transform#, exposure, morphology

def  adj_gaus_blur(img_i, gause_size = 5, file_type = ".jpg"):
    """
    图像高斯模糊函数
    :param: img_i 类型:ndarray 数据
    :param: gause_size 类型:整数  定义:高斯滤波器模板大小
    :param: file_type 类型:字符串 定义:图像文件后缀名 默认.jpg
    return: 增强后图片
    """    
    img_i = cv2.GaussianBlur(img_i, (gause_size, gause_size), 0) #高斯模糊
    
    return img_i
 

def  add_gaus_noise(img_i, sigma = 0.1):
    """
    图像加高斯噪声函数
    :param: img_i 类型:ndarray 数据
    :param: sigma 类型:浮点数 定义:高斯噪声的标准差
    return: 增强后图片
    """
    noise = np.random.normal(0, sigma, img_i.shape) #标准正态噪声
    img_i = img_i / 255 #图像/255再修正
    img_i = noise + img_i
    img_i[img_i > 1] = 1  
    img_i[img_i < 0] = 0  #修正到[0,1]内
    img_i = np.array(img_i * 255, dtype = "uint8" )    
    
    return img_i


def  flip(img_i):
    """
    翻转图像函数
    :param: img_i 类型:ndarray 数据
    return: 增强后图片
    """
    if np.random.randn() < 1/4:
        img_i = cv2.flip(img_i, -1) #翻转
    if np.random.randn() < 1/4:
        img_i = cv2.flip(img_i, 0) #翻转
    if np.random.randn() < 1/4:
        img_i = cv2.flip(img_i, 1) #翻转
        
    return img_i


def add_black_frame(img_i, add_wid_max_rate = 0.1):
    """
    图像加黑边函数
    :param: img_i 类型:ndarray 数据
    :param: add_wid_max 类型:整数 定:黑框宽度上限
    return: 增强后图片
    """ 
    width, length = img_i.shape #图像长宽高
    
    add_wid_max = int(add_wid_max_rate * min(width, length))
    add_shape=[np.random.randint(0, add_wid_max) for _ in range(4)] #最大范围内随机加黑边
    up, down, left, right = add_shape  #四个黑边边框宽度
    black=np.zeros((width + up + down, length + left + right)) #大黑布
    black[up: width + up , left: length + left] = img_i #嵌入图片
                
    return black


def cut_random(img_i, cut_ratio = 0.5):
    """
    随机裁剪函数
    :param: img_i 类型:ndarray 数据
    :param: cut_ratio 类型:浮点数 定义:裁剪长宽比例限定
    return: 增强后图片
    """
    width, length = img_i.shape #图像长宽高
    
    #随机初始点 + 两个边长   两个边长 >= 原图的长宽 * cut_ratio
    start_x, start_y = np.random.randint(0, width * cut_ratio),  np.random.randint(0, length * cut_ratio) #初始点坐标
    side_x, side_y = np.random.randint(width * cut_ratio, width - start_x),  np.random.randint(length * cut_ratio, length - start_y) #两个边长
    
    img_i = img_i[start_x:start_x + side_x, start_y:start_y + side_y] #随机裁剪
    
    return img_i

 
def resize(img_i, resize_size):
    """
    调整大小函数
    :param: img_i 类型:ndarray 数据
    return: 增强后图片
    """
    return transform.resize(img_i, resize_size)


def normalize(img_i):
    """
    正规化函数并调整格式
    :param: img_i 类型:ndarray 数据
    return: 增强后图片
    """
    mean_ = np.mean(img_i)
    std_ = np.std(img_i) + 0.0001
    image_new = (img_i - mean_) / std_
    image_new = np.reshape(image_new, image_new.shape + (1,))
    return image_new
