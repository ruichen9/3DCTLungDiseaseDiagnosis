# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:07:47 2020

@author: 12057
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import scipy.ndimage
import SimpleITK as sitk
#import gdcm
import pydicom
import cv2
import matplotlib.pyplot as plt
import pickle
import glob
import time
import pandas as pd
import pickle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, morphology
#from mayavi import mlab

def glob_allfile(file_holder, file_type):
    """
    获取文件夹下包括子文件夹下的固定类型的所有文件
    :param: file_type 图片后缀名
    :param: file_holder 目标文件夹
    :return: all_file, 所有文件
    """
    all_file = []
    for root, dirs, files in os.walk(file_holder):
        for file in files:
            if os.path.splitext(file)[1] == file_type :
                if file_type == "" and not is_dicom_file(os.path.join(root, file)):
                    continue
                all_file.append(os.path.join(root, file))
    return all_file


def plot_3d(image, threshold = -300, mlab_ = False):    
    #3d图片可视化
    p = image.transpose(2,1,0)
    
    verts, faces, mean, val = measure.marching_cubes_lewiner(p, 0.0)
    #verts, faces = measure.marching_cubes(p, threshold)
    if mlab_:
        mlab.triangular_mesh([vert[0] for vert in verts],
                             [vert[1] for vert in verts],
                             [vert[2] for vert in verts],
                             faces) # doctest: +SKIP
        mlab.show()
    else:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces], alpha = 0.70)
        face_color = [0.45, 0.45, 0.75]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    
        ax.set_xlim(0, p.shape[0])
        ax.set_ylim(0, p.shape[1])
        ax.set_zlim(0, p.shape[2])
    
        plt.show()
    
def plot_sanshitu(image):
    #图片三视图可视化
    h, w, c = image.shape
    wei1 = image[:, :, c // 2]
    wei2 = image[:, w // 2, :]
    wei3 = image[h // 2, :, :]
    
    fig = plt.figure(figsize=(30, 20))
    ax1 = fig.add_subplot(311)
    plt.imshow(wei1)
    ax2 = fig.add_subplot(312)
    plt.imshow(wei2)
    ax3 = fig.add_subplot(313)
    plt.imshow(wei3)
        
    plt.show()

def is_dicom_file(filename):
    '''
       判断某文件是否是dicom格式的文件
    :param filename: dicom文件的路径
    :return:
    '''
    with open(filename, 'rb') as file_stream:
        file_stream.seek(128)
        data = file_stream.read(4)
    if data == b'DICM':
        return True
    return False

def readAllfolders(folder):
    #读取所有包含带slicelocation信息的文件夹。用于提取数据处理
    ddir = []
    for root,dirs,files in os.walk(folder):
        #检查每一个root下面是否有带attr的dicom文件
        dirlist = os.listdir(root)
        for ifile in dirlist:
            if(os.path.isfile(os.path.join(root,ifile))):
                if(is_dicom_file(os.path.join(root,ifile))):
                    TheFile = pydicom.dcmread(os.path.join(root,ifile))
                    if(hasattr(TheFile,'SliceLocation')):
                        ddir.append(root)
    return list(set(ddir))

def readAllsubs(folder):
    #读取所有包含带slicelocation信息的文件名。并进行排序，用于提取数据处理
    ddir = []
    for root,dirs,files in os.walk(folder):
        #检查每一个root下面是否有带attr的dicom文件
        dirlist = os.listdir(root)
        for ifile in dirlist:
            if(os.path.isfile(os.path.join(root,ifile))):
                if(is_dicom_file(os.path.join(root,ifile))):
                    TheFile = pydicom.dcmread(os.path.join(root,ifile))
                    if(hasattr(TheFile,'SliceLocation')) and TheFile.SliceLocation:
                        ddir.append( (os.path.join(root,ifile), TheFile.SliceLocation) )
    ddir.sort(key = lambda x: x[1])
    return ddir

def get_pixels_hu_by_simpleitk(dicom_dir):
    '''
    读取某dicom文件,并提取像素值(-4000 ~ 4000)
    :param src_dir: dicom文件夹路径
    :return: image array
    '''
    itk_img = sitk.ReadImage(dicom_dir)
    img_array = sitk.GetArrayFromImage(itk_img)
    #slope = dicomi.RescaleSlope
    #intercept = dicomi.RescaleIntercept
    #img_array = img_array.astype('float16') / slope - intercept
    
    return img_array[0,:,:].astype('int16')


def get_pixels_hu_by_pydicom(dicom_dir):
    '''
    读取某dicom文件,并提取像素值(-4000 ~ 4000)
    :param src_dir: dicom文件夹路径
    :return: image array
    '''
    dicomi = pydicom.read_file(dicom_dir)    
    img_array = dicomi.pixel_array
    img_array[img_array == -2000] = 0
    slope = dicomi.RescaleSlope
    intercept = dicomi.RescaleIntercept
    img_array = img_array.astype('float16') * slope + intercept
    return img_array.astype('int16')

def center_crop(img):
    h, w = img.shape[0], img.shape[1]
    cen_x, cen_y = h // 2, w // 2
    xmin, xmax = cen_x - 200, cen_x + 200
    ymin, ymax = cen_y - 250, cen_y + 250    
    return img[xmin:xmax, ymin:ymax]
    

def tobgr0_255(img, mid, width):
    error_min = img < -1024 
    error_max = img > 3095
    img = (img - mid + 0.5 * width) * 255 / width
    img = np.clip(img, 0, 255)
    img[error_min | error_max] = 0
    return img

def resample(image, old_spacing, new_spacing = [1,1,1]):
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor  
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image
    
def thick_select(img_list):
    real_slice_nums = len(img_list)
    dic = {}
    for img_path, location in img_list:
        dicomi = pydicom.read_file(img_path)
        dic.setdefault(dicomi.SliceThickness, []).append([img_path, dicomi.PixelSpacing])
    dic = sorted(dic.items(), key = lambda x: len(x[1]))
    if len(dic[-1][1]) > 10:
        print('thickness = {}'.format(dic[-1][0]) + 
              '  slice_nums = {}'.format(len(dic[-1][1])) + 
              '  real_slice_nums = {}'.format(real_slice_nums))

    thickness =  np.array([float(x) for x in dic[-1][1][1][1]] + [float(dic[-1][0])], dtype = np.float32)
    return [x[0] for x in dic[-1][1]], thickness, len(dic[-1][1])
    
def data_get(file_holder, data_dir = 'xinguan2', sanshitu = False, sanwei = True, erwei = True):
    
    if not os.path.exists(os.path.join(file_holder, 'total_data')):
        os.mkdir(os.path.join(file_holder, 'total_data'))
    if not os.path.exists(os.path.join(file_holder, 'total_data', 'xin guan')):
        os.mkdir(os.path.join(file_holder, 'total_data', 'xin guan'))
        os.mkdir(os.path.join(file_holder, 'total_data', 'xin guan', 'img'))
        os.mkdir(os.path.join(file_holder, 'total_data', 'xin guan', 'pkl'))

    if not os.path.exists(os.path.join(file_holder, 'total_data', 'no xinguan')):
        os.mkdir(os.path.join(file_holder, 'total_data', 'no xinguan'))
        os.mkdir(os.path.join(file_holder, 'total_data', 'no xinguan', 'img'))

        os.mkdir(os.path.join(file_holder, 'total_data', 'no xinguan', 'img', 'bingdu'))
        os.mkdir(os.path.join(file_holder, 'total_data', 'no xinguan', 'img', 'xijun'))
        os.mkdir(os.path.join(file_holder, 'total_data', 'no xinguan', 'img', 'jiehe'))
        os.mkdir(os.path.join(file_holder, 'total_data', 'no xinguan', 'img', 'zhengchang'))

        os.mkdir(os.path.join(file_holder, 'total_data', 'no xinguan', 'pkl'))
        os.mkdir(os.path.join(file_holder, 'total_data', 'no xinguan', 'pkl', 'bingdu'))
        os.mkdir(os.path.join(file_holder, 'total_data', 'no xinguan', 'pkl', 'xijun'))
        os.mkdir(os.path.join(file_holder, 'total_data', 'no xinguan', 'pkl', 'jiehe'))
        os.mkdir(os.path.join(file_holder, 'total_data', 'no xinguan', 'pkl', 'zhengchang'))

    if not os.path.exists(os.path.join(file_holder, 'question_data')):
        os.mkdir(os.path.join(file_holder, 'question_data'))
    if not os.path.exists(os.path.join(file_holder, 'question_data', 'xin guan')):
        os.mkdir(os.path.join(file_holder, 'question_data', 'xin guan'))
        os.mkdir(os.path.join(file_holder, 'question_data', 'xin guan', 'img'))
        os.mkdir(os.path.join(file_holder, 'question_data', 'xin guan', 'pkl'))

    if not os.path.exists(os.path.join(file_holder, 'question_data', 'no xinguan')):
        os.mkdir(os.path.join(file_holder, 'question_data', 'no xinguan'))
        os.mkdir(os.path.join(file_holder, 'question_data', 'no xinguan', 'img'))

        os.mkdir(os.path.join(file_holder, 'question_data', 'no xinguan', 'img', 'bingdu'))
        os.mkdir(os.path.join(file_holder, 'question_data', 'no xinguan', 'img', 'xijun'))
        os.mkdir(os.path.join(file_holder, 'question_data', 'no xinguan', 'img', 'jiehe'))
        os.mkdir(os.path.join(file_holder, 'question_data', 'no xinguan', 'img', 'zhengchang'))

        os.mkdir(os.path.join(file_holder, 'question_data', 'no xinguan', 'pkl'))
        os.mkdir(os.path.join(file_holder, 'question_data', 'no xinguan', 'pkl', 'bingdu'))
        os.mkdir(os.path.join(file_holder, 'question_data', 'no xinguan', 'pkl', 'xijun'))
        os.mkdir(os.path.join(file_holder, 'question_data', 'no xinguan', 'pkl', 'jiehe'))
        os.mkdir(os.path.join(file_holder, 'question_data', 'no xinguan', 'pkl', 'zhengchang'))
        
    wronglist = []
    wrong = 0
    
    xinguanpath = os.path.join(file_holder, data_dir, 'xinguan')
    noxinguanpath = os.path.join(file_holder, data_dir, 'noxinguan')
    
    xg_name_list = glob.glob(os.path.join(xinguanpath, '*'))
    nxg_name_list = glob.glob(os.path.join(noxinguanpath, '*', '*'))
    nxg_type_list = ['bingdu', 'jiehe', 'xijun', 'zhengchang']
    data_info = pd.DataFrame(columns = ['type', 'patient_name', 'path', 'ct_time', 'thickness', 'slice_nums', 
                                        'total_slice_nums', 'min_pic_shape', 'max_pic_shape',
                                        'min_pic_value', 'max_pic_value'])
    #avg_gray = 58.8514
    
    for xg_name in xg_name_list:
        ct_time = readAllfolders(xg_name)
        name_patient = os.path.split(xg_name)[1]   
        true_time = 0
        for time_ in range(len(ct_time)):
            ct_name = ct_time[time_]
            img_list = readAllsubs(ct_name)
            real_slice_num = len(img_list)
            if len(img_list) >= 10:
                print(name_patient + '_' + os.path.split(ct_name)[1] + ' :')
                img_list, thickness, slice_nums = thick_select(img_list)
                true_time += 1
                pici = np.zeros((512, 512, len(img_list)))
                shapei = []
                min_pic, max_pic = [], []
                for j in range(len(img_list)):
                    img_path = img_list[j]
                    try:
                        """
                        dicomi = pydicom.read_file(img_path)
                        infoi = {}
                        for lis in dicomi.dir("pat"):
                            infoi[lis] = dicomi[lis]
                        """
                        picij = get_pixels_hu_by_simpleitk(img_path)
                        #picij = get_pixels_hu_by_pydicom(img_path)  
                        shapeij = picij.shape
                        shapei.append(shapeij)
                        min_pic.append(np.min(picij))
                        max_pic.append(np.max(picij))
                        #picij = center_crop(picij)
                        picij = cv2.resize(picij, (512, 512))
                        picij = tobgr0_255(picij, -500, 1000)
                        """
                        alpha = 1 + (np.average(picij[:,:,0]) - avg_gray) / avg_gray / 2
                        picij = picij.astype('float64')
                        picij *= alpha
                        """
                        pici[:, :, j] = picij
                    except: #暂时防爆措施
                        pici[:, :, j] = pici[:, :, j - 1]
                        wronglist.append(img_path)
                        wrong += 1
                        print(img_path)
                        pass
                    
                data_info.loc[len(data_info)] = ['xinguan', ct_name, name_patient, os.path.split(ct_name)[1], thickness, slice_nums, 
                              real_slice_num, min(shapei, key = lambda x: x[0] * x[1]), max(shapei, key = lambda x: x[0] * x[1]),
                              np.min(min_pic), np.max(max_pic)]

                pici = resample(pici, thickness)
                print('')

                h, w, c = pici.shape 
                
                baoliu = 'y'               
                if sanwei:
                    #plot_3d(pici, 400)
                    plot_sanshitu(pici)
                    baoliu = input('是否保留或者待定 y/n/任意:  ')
                    
                dim_num = 13
                for dim in range(1, dim_num - 4):
                    #picdim = np.zeros((512, 512, 3))
                    wei1 = pici[:, :, int(c / (dim_num + 1) * (dim + 3))]
                    wei2 = pici[:, int(w / (dim_num + 1) * (dim + 3)), :]
                    wei3 = pici[int(h / (dim_num + 1) * (dim + 3)), :, :]
                        
                    """
                    wei1 = cv2.resize(wei1, (512, 512))
                    wei2 = cv2.resize(wei2, (512, 512))
                    wei3 = cv2.resize(wei3, (512, 512))
                    picdim[:,:,0] = wei1
                    picdim[:,:,1] = wei2
                    picdim[:,:,2] = wei3
                    """
                    
                    if erwei:
                        #cv2.imwrite(os.path.join(file_holder, 'no xinguan', 'img', name_patient + '_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), picdim)   
                        if  sanshitu:
                            if baoliu in ['y','Y']:
                                cv2.imwrite(os.path.join(file_holder, 'total_data', 'xin guan', 'img', 'xinguan_' + name_patient + '_wei1_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), wei1)   
                                cv2.imwrite(os.path.join(file_holder, 'total_data', 'xin guan', 'img', 'xinguan_' + name_patient + '_wei2_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), wei2)   
                                cv2.imwrite(os.path.join(file_holder, 'total_data', 'xin guan', 'img', 'xinguan_' +  name_patient + '_wei3_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), wei3)   
                            elif baoliu not in ['n','N']:
                                cv2.imwrite(os.path.join(file_holder, 'question_data', 'xin guan', 'img', 'xinguan_' + name_patient + '_wei1_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), wei1)   
                                cv2.imwrite(os.path.join(file_holder, 'question_data', 'xin guan', 'img', 'xinguan_' + name_patient + '_wei2_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), wei2)   
                                cv2.imwrite(os.path.join(file_holder, 'question_data', 'xin guan', 'img', 'xinguan_' +  name_patient + '_wei3_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), wei3)   

                        else:
                            if baoliu in ['y','Y']:
                                cv2.imwrite(os.path.join(file_holder, 'total_data', 'xin guan', 'img', 'xinguan_' + name_patient + '_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), wei1)   
                            elif baoliu not in ['n','N']:
                                cv2.imwrite(os.path.join(file_holder, 'question_data', 'xin guan', 'img', 'xinguan_' + name_patient + '_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), wei1)   

                if sanwei:
                    if baoliu in ['y','Y']:
                        with open(os.path.join(file_holder, 'total_data', 'xin guan', 'pkl', 'xinguan_' + name_patient + '_' +  os.path.split(ct_name)[1] + '.pkl'), 'wb') as pkl:
                            pickle.dump(pici, pkl)
                    elif baoliu not in ['n','N']:
                        with open(os.path.join(file_holder, 'question_data', 'xin guan', 'pkl', 'xinguan_' + name_patient + '_' +  os.path.split(ct_name)[1] + '.pkl'), 'wb') as pkl:
                            pickle.dump(pici, pkl)


    for nxg_name in nxg_name_list:
        ct_time = readAllfolders(nxg_name)
        name_patient = os.path.split(nxg_name)[1] 
        true_time = 0
        for time_ in range(len(ct_time)):
            ct_name = ct_time[time_]
            img_list = readAllsubs(ct_name)
            real_slice_num = len(img_list)
            if len(img_list) >= 10:
                true_time += 1
                for nxg_type in nxg_type_list:
                    if nxg_type in ct_name:
                        print(nxg_type + '_' + name_patient + '_' + os.path.split(ct_name)[1] + ' :')
                        break
                img_list, thickness, slice_nums = thick_select(img_list)
                pici = np.zeros((512, 512, len(img_list)))
                shapei = []
                min_pic, max_pic = [], []
                for j in range(len(img_list)):
                    img_path = img_list[j]
                    try:
                        """
                        dicomi = pydicom.read_file(img_path)
                        infoi = {}
                        for lis in dicomi.dir("pat"):
                            infoi[lis] = dicomi[lis]
                        """
                        picij = get_pixels_hu_by_simpleitk(img_path)
                        #picij = get_pixels_hu_by_pydicom(img_path)                                            
                        #picij = center_crop(picij)
                        shapeij = picij.shape
                        shapei.append(shapeij)
                        min_pic.append(np.min(picij))
                        max_pic.append(np.max(picij))
                        picij = cv2.resize(picij, (512, 512))
                        picij = tobgr0_255(picij, -500, 1000)
                        """
                        alpha = 1 + (np.average(picij[:,:,0]) - avg_gray) / avg_gray / 2
                        picij = picij.astype('float64')
                        picij *= alpha
                        """
                        pici[:, :, j] = picij
                    except: #暂时防爆措施
                        pici[:, :, j] = pici[:, :, j - 1]
                        wronglist.append(img_path)
                        wrong += 1
                        print(img_path)
                        pass
                    
                data_info.loc[len(data_info)] = [nxg_type, ct_name, name_patient, os.path.split(ct_name)[1], thickness, slice_nums, 
                              real_slice_num, min(shapei, key = lambda x: x[0] * x[1]), max(shapei, key = lambda x: x[0] * x[1]),
                              np.min(min_pic), np.max(max_pic)]
                
                pici = resample(pici, thickness)
                print('')
                h, w, c = pici.shape
                
                baoliu = 'y'               
                if sanwei:
                    #plot_3d(pici, 400)
                    plot_sanshitu(pici)
                    baoliu = input('是否保留或者待定 y/n/任意:  ')
                    
                dim_num = 13
                for dim in range(1, dim_num - 4):
                    #picdim = np.zeros((512, 512, 3))
                    wei1 = pici[:, :, int(c / (dim_num + 1) * (dim + 3))]
                    wei2 = pici[:, int(w / (dim_num + 1) * (dim + 3)), :]
                    wei3 = pici[int(h / (dim_num + 1) * (dim + 3)), :, :]
                        
                    """
                    wei1 = cv2.resize(wei1, (512, 512))
                    wei2 = cv2.resize(wei2, (512, 512))
                    wei3 = cv2.resize(wei3, (512, 512))
                    picdim[:,:,0] = wei1
                    picdim[:,:,1] = wei2
                    picdim[:,:,2] = wei3
                    """
                    
                    if erwei:
                        #cv2.imwrite(os.path.join(file_holder, 'no xinguan', 'img', name_patient + '_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), picdim)   
                        if  sanshitu:
                            if baoliu in ['y','Y']:
                                cv2.imwrite(os.path.join(file_holder, 'total_data', 'no xinguan', 'img', nxg_type, name_patient + '_wei1_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), wei1)   
                                cv2.imwrite(os.path.join(file_holder, 'total_data', 'no xinguan', 'img', nxg_type, name_patient + '_wei2_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), wei2)   
                                cv2.imwrite(os.path.join(file_holder, 'total_data', 'no xinguan', 'img', nxg_type, name_patient + '_wei3_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), wei3)   
                            elif baoliu not in ['n','N']:
                                cv2.imwrite(os.path.join(file_holder, 'question_data', 'no xinguan', 'img', nxg_type, name_patient + '_wei1_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), wei1)   
                                cv2.imwrite(os.path.join(file_holder, 'question_data', 'no xinguan', 'img', nxg_type, name_patient + '_wei2_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), wei2)   
                                cv2.imwrite(os.path.join(file_holder, 'question_data', 'no xinguan', 'img', nxg_type,  name_patient + '_wei3_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), wei3)   

                        else:
                            if baoliu in ['y','Y']:
                                cv2.imwrite(os.path.join(file_holder, 'total_data', 'no xinguan', 'img', nxg_type, name_patient + '_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), wei1)   
                            elif baoliu not in ['n','N']:
                                cv2.imwrite(os.path.join(file_holder, 'question_data', 'no xinguan', 'img', nxg_type, name_patient + '_' + os.path.split(ct_name)[1] + '_' + str(dim) + '.png'), wei1)   

                if sanwei:
                    if baoliu in ['y','Y']:
                        with open(os.path.join(file_holder, 'total_data', 'no xinguan', 'pkl', nxg_type, name_patient + '_' +  os.path.split(ct_name)[1] + '.pkl'), 'wb') as pkl:
                            pickle.dump(pici, pkl)
                    elif baoliu not in ['n','N']:
                        with open(os.path.join(file_holder, 'question_data', 'no xinguan', 'pkl', nxg_type, + name_patient + '_' +  os.path.split(ct_name)[1] + '.pkl'), 'wb') as pkl:
                            pickle.dump(pici, pkl)
    data_info.to_excel(os.path.join(file_holder, 'total_data', 'data_info.xlsx'), index = False)
    
    return wrong, wronglist

def gray_sub(img):
    img_array = img.ravel()
    np.percentile(img_array, 0.75) - np.percentile(img_array, 0.25)
    
def avg_gray_value(img_list):
    avg_gray = []
    for img_path in img_list:
        imgi = cv2.imread(img_path)[:,:,0]
        avg_gray.append( np.average(imgi) )
    avg_gray.sort()
    return avg_gray[len(avg_gray) // 2]
#58.8514
        
def data_contrast(file_holder):
    img_list = []
    img_list += glob_allfile(os.path.join(file_holder, 'total_data', 'xin guan', 'img'), '.png')
    img_list += glob_allfile(os.path.join(file_holder, 'total_data', 'no xinguan', 'img'), '.png')
    
    xishu = []
    avg_gray = avg_gray_value(img_list)
    for img_path in img_list:
        imgi = cv2.imread(img_path)
        alpha = 1 + (np.average(imgi[:,:,0]) - avg_gray) / avg_gray / 2
        xishu.append(alpha)
        imgi0 = imgi[:,:,0].astype('float64')
        imgi0 *= alpha
        imgi0 = imgi.astype('uint8')
        imgi0 = np.clip(imgi0, 0, 255)
        imgi[:,:,0] = imgi0
        cv2.imwrite(img_path, imgi)
    return xishu 

    
def main(file_holder = r'./', data_dir = 'xinguan'):
    tic = time.time()
    #file_holder = r'D:\Postgraduate\AI lesson\class project'
    wrong, wronglist = data_get(file_holder, data_dir)
    toc = time.time()
    print('bug_pic_num = {}'.format(wrong))
    print('time = {}s'.format(toc - tic))
    #xishu = data_contrast(file_holder)
    
    return wrong, wronglist#, xishu
    
if __name__ == '__main__':
    wrong, wronglist = main(r'./', 'test_data')