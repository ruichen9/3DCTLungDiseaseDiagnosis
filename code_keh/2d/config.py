# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 01:03:15 2020

@author: 12057
"""
import os
import yaml

class Config:
    """
    生成配置类
    """
    def __init__(self, config_path):
        """
        输入config路径, 给参数设置默认值
        """
        print('')
        print('>' * 80)
        print('Config loading begin')
        print('')
        
        self.config_path = config_path #config文件路径
        
        #设置默认值
        self.distribution = False #分布式训练
        self.device = "cpu" #设备选择
        self.data_info = 'CIFAR10' #'CIFAR10', 或者自己文件夹路径
        self.pretrain = None #预训练路径
        self.model_name = 'LeNet' #模型名字
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') #类别
        self.classes_num = len(self.classes) #类个数        
        self.epoch_num = 100 #epoch数
        self.batch_size = 128 #batch大小
        self.cut_scale = (0.2, 1.2) #图片裁剪面积比例范围
        self.cut_ratio = (0.75, 1.3333333333333333) #图片裁剪长宽比比例范围
        self.resize_size = (224, 224) #图片resize大小
        self.mix_up = False #是否mix_up
        self.attention = False #是否attention
        self.momentum = 0.9 #动量
        self.alpha = 0.001 #初始学习率
        self.weight_decay = 0.0001 #权重衰减系数
        self.lr_step = 20 #学习率衰减步数
        self.lr_decay = 0.2 #学习率衰减率
        self.save_path = r'./' #模型保存路径

        if self.load_cfg(): #载入config文件参数
            print('Config loading is valid')
            
        else:
            print('Config loading is invalid')
        print('>' * 80)
        print(' ')      
        print(' ')      
       
    def load_cfg(self):
        """
        载入config参数
        :return:
        """    
        if not os.path.exists(self.config_path): #如果没有配置文件, 不载入参数
            print('No config file, use default param')
            return  False

        # 读取配置文件
        yml_str = open(self.config_path).read()
        try:
            cfg_info = yaml.load(yml_str, Loader = yaml.FullLoader)
            if cfg_info['distribution'] != 0:
                self.distribution = [int(x.strip()) for x in cfg_info['distribution'].split(',')]#分布式
            self.device = cfg_info['device'] #设备选择
           
            if cfg_info['data_info'] != 'CIFAR10':
                print('Data:', cfg_info['data_info'].strip('./'))
                print('')
                assert os.path.exists(cfg_info['data_info']) #检验路径是否存在
            self.data_info = cfg_info['data_info']#'CIFAR10', 或者自己文件夹路径
            
            assert os.path.exists(cfg_info['classes_path']) #检验路径是否存在
            self.classes_path = cfg_info['classes_path'] #类别路径
            
            assert self.get_classes() #获取类别
            
            self.classes_num = cfg_info['classes_num']
            
            self.pretrain = cfg_info['pretrain'] #预训练路径
            if self.pretrain:
                assert os.path.exists(self.pretrain) #检验路径是否存在
                
            self.model_name = cfg_info['model_name'] #模型名字
            self.epoch_num = cfg_info['epoch_num'] #epoch数
            self.batch_size = int(cfg_info['batch_size']) #batch大小
            self.cut_scale = [float(x.strip()) for x in cfg_info['cut_scale'].split(',')] #图片裁剪面积比例范围
            self.cut_ratio = [float(x.strip()) for x in cfg_info['cut_ratio'].split(',')] #图片裁剪长宽比比例范围   
            self.resize_size = [int(x.strip()) for x in cfg_info['resize_size'].split(',')] #图片resize大小
            self.mix_up = cfg_info['mix_up'] #是否mix_up
            self.attention = cfg_info['attention'] #是否attention
            self.momentum = float(cfg_info['momentum']) #动量
            self.alpha = float(cfg_info['alpha']) #初始学习率
            self.weight_decay = float(cfg_info['weight_decay']) #权重衰减系数
            self.lr_step = int(cfg_info['lr_step']) #学习率衰减步数
            self.lr_decay = float(cfg_info['lr_decay']) #学习率衰减率
            
            assert os.path.exists(cfg_info['save_path']) #检验路径是否存在
            self.save_path = cfg_info['save_path'] #模型保存路径
            
            print('Succeed to read config file')
            
            return True
        
        except:
            print('Fail to read config file, use default param')
            
            return False
    
    def get_classes(self):
        """
        载入类别list
        :return:
        """  
        classes = []
        try:
            with open(self.classes_path, 'r') as cls_txt:
                for line in cls_txt.readlines(): # 整行读取数据:
                    line = line.strip('\n') #去除换行符号
                    if not line: #读取完
                        break
                    else:
                        classes.append(line)
            self.classes = classes
            print('Succeed to read classes file')
            
            return True
        
        except:
            print('Fail to read classes file, use default classes')
            
            return False

                        