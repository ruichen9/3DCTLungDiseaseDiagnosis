# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:33:56 2020

@author: 12057
"""
import os
import cv2
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform#, exposure, morphology


"""
3d随机旋转
"""
def rotate_3d(image3d):
    if np.random.rand() < 0.5:      
        h,w,c = image3d.shape
        theta = np.random.randint(-30, 30)
        phi = np.random.randint(-30, 30)
        gamma = np.random.randint(-30, 30)
        
        dx = theta / 180 * (h // 2)
        dy = phi / 180 * (w // 2)
        dz = gamma / 180 * (c // 2)
    
        #image3d = np.apply_over_axes(rotate_2d(theta, phi, gamma, dx, dy, dz), image3d, (0,1))
        img_rotate = np.zeros(image3d.shape)
        for i in range(c):
            img_rotate[:,:,i] = rotate_along_axis(image3d[:,:,i], theta, phi, gamma, dx, dy, dz)
        
        return img_rotate
    else:
        return image3d

#def rotate_2d(theta = 0, phi = 0, gamma = 0, dx = 0, dy = 0, dz = 0):
    
def rotate_along_axis(image, theta = 0, phi = 0, gamma = 0, dx = 0, dy = 0, dz = 0):

    # Get radius of rotation along 3 axes
    height, width = image.shape[0], image.shape[1]
    rtheta, rphi, rgamma = theta * pi / 180, phi * pi / 180, gamma * pi / 180
    
    # Get ideal focal length on z axis
    # NOTE: Change this section to other axis if needed
    d = np.sqrt(height ** 2 + width ** 2)
    focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
    dz = focal
    
    # Get projection matrix
    mat = get_M(width, height, focal, rtheta, rphi, rgamma, dx, dy, dz)
    
    return cv2.warpPerspective(image.copy(), mat, (width, height))
    
#return rotate_along_axis(theta, phi, gamma, dx, dy, dz)

""" Get Perspective Projection Matrix """
def get_M(width, height, focal, theta, phi, gamma, dx, dy, dz):

    w = width
    h = height
    f = focal
    
    # Projection 2D -> 3D matrix
    A1 = np.array([ [1, 0, -w/2],
                    [0, 1, -h/2],
                    [0, 0, 1],
                    [0, 0, 1]])
    
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([ [1, 0, 0, 0],
                    [0, np.cos(theta), -np.sin(theta), 0],
                    [0, np.sin(theta), np.cos(theta), 0],
                    [0, 0, 0, 1]])
    
    RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                    [0, 1, 0, 0],
                    [np.sin(phi), 0, np.cos(phi), 0],
                    [0, 0, 0, 1]])
    
    RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                    [np.sin(gamma), np.cos(gamma), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    
    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)
    
    # Translation matrix
    T = np.array([  [1, 0, 0, dx],
                    [0, 1, 0, dy],
                    [0, 0, 1, dz],
                    [0, 0, 0, 1]])
    
    # Projection 3D -> 2D matrix
    A2 = np.array([ [f, 0, w/2, 0],
                    [0, f, h/2, 0],
                    [0, 0, 1, 0]])
    
    # Final transformation matrix
    return np.dot(A2, np.dot(T, np.dot(R, A1)))
    

"""
3d随机裁剪+还原大小
""" 
def cut_3d(image3d):
    h,w,c = image3d.shape
    if np.random.rand() < 0.5:      
        h_end = np.random.randint(int(0.9 * h), h)
        w_end = np.random.randint(int(0.9 * w), w)
        c_end = np.random.randint(int(0.9 * c), c)
        
        h_start = np.random.randint(0, int(0.1 * h))
        w_start = np.random.randint(0, int(0.1 * w))
        c_start = np.random.randint(0, int(0.1 * c))
        
        img_cut = image3d[h_start:h_end, w_start:w_end, c_start:c_end]
        #img_cut = transform.resize(img_cut, (h,w,c))
        
        return img_cut
    else:
        return image3d

"""
3d随机翻转
"""
def flip_3d(image3d):
    if np.random.randn() < 1/4:
        image3d = cv2.flip(image3d, 0)
    if np.random.randn() < 1/4:
        image3d = cv2.flip(image3d, 1)
    if np.random.randn() < 1/4:
        image3d = image3d[:,:,::-1]
    return image3d

"""
3d加黑边,还原大小
"""
def add_black_3d(image3d):
    if np.random.rand() < 0.5:      
        h,w,c = image3d.shape
        up, down = np.random.randint(0, int(0.1 * h)), np.random.randint(0, int(0.1 * h))
#         up,down = max([up,down]),min([up,down])
        left, right = np.random.randint(0, int(0.1 * w)), np.random.randint(0, int(0.1 * w))
        above, below = np.random.randint(0, int(0.1 * c)), np.random.randint(0, int(0.1 * c))
        image_new = np.zeros((h + up + down, w + left + right, c + above + below))
        image_new[up: h + up, left: w + left, above: c + above] = image3d
        #image_new = transform.resize(image_new, (h,w,c))
        return image_new
    else:
        return image3d
    
"""
正规化
"""
def normalizeOld(image3d):
    mean_ = np.mean(image3d, axis = (0, 1))
    std_ = np.std(image3d, axis = (0, 1)) + 0.0001
    image_new = (image3d - mean_) / std_
    return image_new


"""
图片做到一样大小
"""
def Padding3d(img,target_dim = [700,512,512]):
    x,y,z=img.shape
    x_change = target_dim[0]-x
    y_change = target_dim[1]-y
    z_change = target_dim[2]-z
    if x_change%2==0:
        x_before = int(x_change/2)
        x_after = int(x_change/2)
    else:
        x_before = int((x_change-1)/2)
        x_after = int(x_before + 1)
    if y_change%2==0:
        y_before = int(y_change/2)
        y_after = int(y_change/2)
    else:
        y_before = int((y_change-1)/2)
        y_after = int(y_before + 1)
    if z_change%2==0:
        z_before = int(z_change/2)
        z_after = int(z_change/2)
    else:
        z_before = int((z_change-1)/2)
        z_after = int(z_before + 1)
    result = np.pad( img, pad_width = ((x_before,x_after),(y_before,y_after),(z_before,z_after)) ,constant_values=((0,0),(0,0),(0,0)))
    return result
"""
开窗
"""
def tobgr0_255(image3d, mid, width):
    error_min = image3d < -1024 
    error_max = image3d > 3095
    image3d = (image3d - mid + 0.5 * width) * 255 / width
    image3d = np.clip(image3d, 0, 255)
    image3d[error_min | error_max] = 0
    image3d = image3d.astype('uint8')
    return image3d

"""
转换DHW -> HWD
"""
def tohwd(image3d):
    return np.transpose(image3d, (1,2,0))
    
"""
调整大小
"""  
def resize_size(image3d, resize_size):
    return transform.resize(image3d, resize_size)


    
def normalize(image, MIN_BOUND = -1000.0, MAX_BOUND = 400.0):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image



def zero_center(image, PIXEL_MEAN = 0.25):
    image = image - PIXEL_MEAN
    return image

def Padding3d(img,target_dim = [700,512,512]):
    x,y,z=img.shape
    x_change = target_dim[0]-x
    y_change = target_dim[1]-y
    z_change = target_dim[2]-z
    if x_change%2==0:
        x_before = int(x_change/2)
        x_after = int(x_change/2)
    else:
        x_before = int((x_change-1)/2)
        x_after = int(x_before + 1)
    if y_change%2==0:
        y_before = int(y_change/2)
        y_after = int(y_change/2)
    else:
        y_before = int((y_change-1)/2)
        y_after = int(y_before + 1)
    if z_change%2==0:
        z_before = int(z_change/2)
        z_after = int(z_change/2)
    else:
        z_before = int((z_change-1)/2)
        z_after = int(z_before + 1)
    result = np.pad( img, pad_width = ((x_before,x_after),(y_before,y_after),(z_before,z_after)) ,constant_values=((0,0),(0,0),(0,0)))
    return result
