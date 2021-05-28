# To add a new cell, type '##'
# To add a new markdown cell, type '## [markdown]'
##

import os, tarfile
import numpy as np
import os
import glob
# import SimpleITK as sitk
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import pydicom
import sys
import pandas as pd
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy


##
def un_targz(source_file):
    f = tarfile.open(source_file)
    names = f.getnames()
    targetname=source_file.replace(".tgz","")
    for name in names:
        f.extract(name, path=targetname)


##
def removeFiles(FileList, fileType = ".npy"):
    for i in range(len(FileList)):
        if FileList[i].endswith(fileType):
            os.remove(FileList[i])


##
if 0:
    removeFiles(FileList, fileType = ".npy")
    FileList = os.listdir("./")
    len(FileList)
    removeFiles(FileList, fileType = ".png")


##
os.getcwd()


##
FileList = os.listdir("./")
len(FileList)
if 0:
    for i in range(len(FileList)):
        if FileList[i].endswith(".tgz"):
            print(i)
            un_targz(FileList[i])



def npy_name(foldername):
    npyname = foldername.replace("D/","D-").replace(":","").replace("市三院CT图像NCP","ISCOVID").replace("/","-").replace("\\","-").replace("正常病毒性肺炎","BINGDU").replace("正常细菌性肺炎","XIJUN").replace("正常结核","JIEHE").replace("正常全阴性","YINXING") + ".npy"
    npyname = "".join(pypinyin.lazy_pinyin(npyname))
    return npyname


##
import os
import cv2
import numpy as np
#from Draw2DFrom3D import *
import pypinyin
import time


##
def isDicomFile(filename):
    with open(filename, 'rb') as file_stream:
        file_stream.seek(128)
        data = file_stream.read(4)
    if data == b'DICM':
        return True
    return False

# file1= "/home/ddata/新冠/2020.01.19/DICOMOBJ/SYN00451"
# is_dicom_file(file1)
# hasattr(f, 'SliceLocation')

def readAllsubs(folder):#读取所有包含带slicelocation信息的文件夹。用于提取数据处理
    ddir = []
    for root,dirs,files in os.walk(folder):#检查每一个root下面是否有带attr的dicom文件
        dirlist = os.listdir(root)
        for ifile in dirlist:
            if(os.path.isfile(os.path.join(root,ifile))):
                if(isDicomFile(os.path.join(root,ifile))):
                    TheFile = pydicom.dcmread(os.path.join(root,ifile))
                    if(hasattr(TheFile,'SliceLocation')):
                        ddir.append(root)
    return list(set(ddir))


def FileNameToFile(sstr):
    return sstr.replace("-","/").replace("G/","G:/").replace(".png","")



def save_npy(image, foldername):
    npyname = foldername.replace("D/","D-").replace(":","").replace("市三院CT图像NCP","ISCOVID").replace("/","-").replace("\\","-").replace("正常病毒性肺炎","BINGDU").replace("正常细菌性肺炎","XIJUN").replace("正常结核","JIEHE").replace("正常全阴性","YINXING") + ".npy"
    npyname = "".join(pypinyin.lazy_pinyin(npyname))
    np.save(npyname, image, allow_pickle=True, fix_imports=True)

def ToLung_npy2(myimage, D,path1):
    result = D
    save_npy(result, path1)

a = np.zeros((3,3,3))
np.pad(a,pad_width=((3,3),(3,3),(3,3)),mode = 'constant', constant_values=0)
def _cut_img(img):
    """
    图像裁剪出肺部
    :param: img 类型:uint8 定义:图像数据
    return: img_cut 类型:uint8 定义:裁剪肺部后图像数据
    """
    # 灰度处理
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    img_resize = cv2.resize(img, (h*2 , w*2 )) #放大图片特征
    #img_resize = img #放大图片特征
    #plt.imshow(img_resize)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    erosion =  cv2.erode(img_resize, kernel_erode, iterations = 8) #图片腐蚀特征
    #plt.imshow(erosion)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    expand = cv2.dilate(erosion, kernel_dilate, iterations = 5)  #膨胀
    #plt.imshow(expand)
    #binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 0) # 二值化(自适应)
    #mean = np.sum(expand) / (h  * w)  #获取均值
    mean = np.sum(expand) / (h  * 4 * w)
    ret, binary =  cv2.threshold(expand, mean, 255, cv2.THRESH_BINARY) #自定义均值二值化
    #plt.imshow(binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contours, key = cv2.contourArea, reverse = True)[0] #获取最大联通域
    # 提取目标区域
    background = np.zeros(binary.shape, np.uint8)
    target_area = cv2.fillPoly(background, [contour], (255,255,255))   
    #plt.imshow(background)
    img_resize[background == 0] = 0
    #plt.imshow(img_resize)    
    return cv2.resize(img_resize, (h,w))
    #return img_resize


def tobgr0_255(img, mid, width):
    image = img.copy()
    error_min = image < -1024 
    error_max = image > 3095
    image = (image - mid + 0.5 * width) * 255 / width
    image = np.clip(image, 0, 255)
    image[error_min | error_max] = 0
    image = image.astype('uint8')
    return image

def ImageNameToFolderName(ImageList):
    N = len(ImageList)
    FolderList = list()
    for i in range(N):
        ImageName = ImageList[i]
        FolderList.append(ImageName.replace("BINGDU","正常病毒性肺炎").replace("JIEHU","正常结核").replace("XIJUN","正常细菌性肺炎").replace("YINXING","正常全阴性").replace("G-COVID_Zhu","G:-COVID_Zhu").replace("-","/").replace("D/","D-").replace(".png",""))
    return FolderList


def arrangeSlice(folderpath):
    files = []
    ppath=os.path.join(folderpath,"*")
    #print('glob: {}'.format(ppath ))
    for fname in glob.glob(ppath, recursive=False):
        #print("loading: {}".format(fname))
        if(os.path.isfile(fname)):
            if(isDicomFile(fname)):
                files.append(pydicom.dcmread(fname))
    #print("file count: {}".format(len(files)))
    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, 'SliceLocation'):
            # print(f['SliceLocation'])
            # print(f['SliceThickness'])
            slices.append(f)
        else:
            skipcount = skipcount + 1
    #print("skipped, no SliceLocation: {}".format(skipcount))
    try:
        slices = sorted(slices, key=lambda s: s.SliceLocation)
    except:
        slices=slices
        #print(folderpath)
    return slices

def get_pixels_hu(slices):
    N=len(slices)
    #assert the slice qualification
    list_slicelocation=[s.SliceLocation for s in slices]
    list_slicethickness=[s.SliceThickness for s in slices]
    listmin=[np.sum(s.pixel_array-s.RescaleIntercept<-1024) for s in slices]
    listmax=[np.sum(s.pixel_array-s.RescaleIntercept>-3096) for s in slices]
    # if 1:
    #     assert(len(list_slicelocation)==len(set(list_slicelocation)))
    #     assert(len(set(list_slicethickness))==1)
    # if 1:
    #sshape=(512,512)
    #image = np.stack([cv2.resize(s.pixel_array,sshape) for s in slices])
    image = np.stack([s.pixel_array for s in slices])
    # if 0:
        # image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    # image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + [scan[0].PixelSpacing[0],scan[0].PixelSpacing[1]], dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

    
def save_3d(image,foldername,threshold=400):
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])  
    ax.set_zlim(0, p.shape[2])
    pngname = foldername.replace(":","").replace("\\","-").replace("/","-")+".png"
    plt.savefig(pngname)
    

def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
    return binary_image


 


##
def RemoveEquipment(myimage):
    n0 = myimage.shape[0]
    result = np.zeros(myimage.shape)
    for i in range(n0):
        a = myimage[i,:,:].copy()
        ajudge = tobgr0_255(a,-700,1500)
        ajudge = np.array(ajudge,np.uint8)
        ajudge = _cut_img(ajudge)
        a[ajudge==0] = 0
        result[i,:,:] = a
    return result


##
def draw3DLinearTrans(image,HU_mean=-1200,HU_Diameter=1000):
    N=len(image)
    HU_upper = HU_mean + HU_Diameter*0.5
    HU_lower = HU_mean - HU_Diameter*0.5
    #assert the slice qualification
    for slice_number in range(N):
        image[slice_number][image[slice_number]>3095]=0
        image[slice_number] = image[slice_number]*(image[slice_number]>HU_lower)*(image[slice_number]<HU_upper) + HU_upper*(image[slice_number]>HU_upper) + HU_lower*(image[slice_number]<HU_lower)
        #image[slice_number]=(image[slice_number] - HU_lower)/(HU_upper-HU_lower)*255
    return image,HU_lower,HU_upper


##




# import pandas as pd
# test=pd.DataFrame(FolderList)
# test.to_csv('TBDcsv.csv',encoding='utf-8',index= None)

def ExtractLungRoundFinal(sourceFolder):
    FolderList = os.listdir(sourceFolder)
    Nowlist = os.listdir("./")
    for i in range(len(FolderList)):
        path1 = FolderList[i]
        # path1 = "G-DataProcessing-RawData-ISCOVID1014_1025-1021tuXsheng-2020_02_01-302_Chest1_25mmStnd.png"
        print(i)
        if not ((path1.replace(":","").replace("\\","-").replace("/","-")) in Nowlist):
            print(path1)
            path1 = FileNameToFile(path1)
            myslices = arrangeSlice(path1)
            myimage2 = get_pixels_hu(myslices)
            myimage, u1,u2 = draw3DLinearTrans(myimage2,HU_mean=-700,HU_Diameter=1000)
            print("begin remove")
            print(time.localtime(time.time()))
            if myimage.shape[0]>50:
                myimage, new_spacing = resample(myimage, myslices, new_spacing=[1,1,1])
                myimage = RemoveEquipment(myimage)
                segmented_lungs_fill = segment_lung_mask(myimage, True)
                save_3d(segmented_lungs_fill*255,path1,threshold = 0)
                plt.close()
                print(time.localtime(time.time()))
                ToLung_npy2(myimage, segmented_lungs_fill, path1)
                print("complete")




sourceFolder = "/data/yeheng/DataProcessing/FromSelectedSampleTonpy/AllRound2_Pickup/AllRound2PNG/AllRound2NoUse"
ExtractLungRoundFinal(sourceFolder)


