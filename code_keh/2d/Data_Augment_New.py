# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 18:15:31 2020

@author: 12057
"""

from PIL import Image

import random
import numpy as np

#test_img = Image.open(r'.\test.jpg')

#自定义数据增强方法, 对PIL Image

#必须切边的裁剪, 针对边界特征
def sidecrop(img, width_range = (0.6, 1.1), height_range = (0.6, 1.1)):
    """
    自定义数据增强 裁图(必须包含边) 
    :param: img: 输入的图像 PIL.Image格式
    :param: width_range 截取图像宽比例的范围，类型为元组 最大值可以大于1,即有概率全取
    :param: height_range: 截取图像的长比例的范围，类型为元组 最大值可以大于1,即全取
    :return: img_sidecrop 截取后的图像
    """
    width, height = img.size[:2] #图片宽长

    width_rate = random.uniform(width_range[0], width_range[1]) #大小对应边缩放比例
    height_rate = random.uniform(height_range[0], height_range[1]) #宽高对应边缩放比例
    
    if width_rate > 1 and height_rate > 1:#防止取原图
        width_rate -= (width_range[1] - 1)
        height_rate -= (height_range[1] - 1)  #减少到 < 1     

    crop_width = width_rate * width if width_rate < 1 else width #裁剪的图片宽度，比例超过1全取
    crop_height = height_rate * height if height_rate < 1 else height #裁剪的图片高度，比例超过1全取
    
    x = 0 if random.random() < 0.5 else width - crop_width #角点随机设置在上或者下
    y = 0 if random.random() < 0.5 else height - crop_height #xy角点随机设置在上或者下
    
    img_sidecrop = img.crop((x, y, x + crop_width, y + crop_height)) #裁图
    
    return img_sidecrop


#图片随机加黑边，针对边框背景
def addblack(img, add_wid_max_rate = 0.2):
    """
    自定义数据增强 裁图(必须包含边) 
    :param: img: 输入的图像 PIL.Image格式
    :param: add_wid_max_rate 黑边最大宽度百分比
    :return: img_addblack 加黑边后的图像
    """
    width, height = img.size[:2] #图像宽长

    add_wid_max = add_wid_max_rate * min(width, height) #黑边最大宽度
    add_shape = [np.random.randint(-add_wid_max, add_wid_max) for _ in range(4)] #最大范围内随机加黑边, 小于零意味不加边
    add_shape = [add if add > 0 else 0 for add in add_shape] #小于零部分不加边
    if add_shape == [0, 0, 0, 0]: #防止输出原图
        add_shape = [np.random.randint(0, add_wid_max) for _ in range(4)]
    up, down, left, right = add_shape  #四个黑边边框宽度
    img_addblack = Image.new(img.mode, (width + up + down, height + left + right), 0) #新建大黑布
                    #模式 如RGB              尺寸 2维tuple                  颜色，可以元组或者数字，默认全黑
    img_addblack.paste(img, (left, up)) #粘贴图片
    
    return img_addblack
    