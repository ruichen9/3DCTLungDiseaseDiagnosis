# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:15:12 2020

@author: 12057
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 18:11:40 2020

@author: 12057
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:18:38 2020

@author: 12057
"""
from __future__ import print_function
import sys
sys.path.append(r'/home/sufedc_nvidia_newgyh/apex')
from apex import amp

import torch
from torch.autograd import Variable
import torch.nn as nn #类
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import math
import os
import time
import numpy as np
import skimage.io as io
import cv2
import PIL

import warnings
warnings.filterwarnings("ignore")

from config import Config #获取配置
from DAE_data_loader import data_get

class Denoising_AutoEncoder(nn.Module):
    """
    autoencoder神经网络结构搭建
    """
    cfg_conv = [[3, 64, 3, 1, 1], [64, 64, 3, 1, 1], 'M', [64, 128, 3, 1, 1], [128, 128, 3, 1, 1], 'M',
                 [128, 256, 3, 1, 1], [256, 256, 3, 1, 1], [256, 256, 3, 1, 1], [256, 256, 3, 1, 1], 'M', 
                 [256, 512, 3, 1, 1], [512, 512, 3, 1, 1], [512, 512, 3, 1, 1], [512, 512, 3, 1, 1], 'M', 
                 [512, 512, 3, 1, 1], [512, 512, 3, 1, 1], [512, 512, 3, 1, 1], [512, 512, 3, 1, 1], 'M']
    #encode网络架构图  conv:[inner_channels, outer_channels, filter, stride, padding]
                      #最大池化层:M  2*2 步长2

    cfg_tranconv = ['M', [512, 512, 3, 1, 1], [512, 512, 3, 1, 1], [512, 512, 3, 1, 1], [512, 512, 3, 1, 1],
                    'M', [512, 512, 3, 1, 1], [512, 512, 3, 1, 1], [512, 512, 3, 1, 1], [512, 256, 3, 1, 1],  
                    'M', [256, 256, 3, 1, 1], [256, 256, 3, 1, 1], [256, 256, 3, 1, 1],  [256, 128, 3, 1, 1],
                    'M', [128, 128, 3, 1, 1], [128, 64, 3, 1, 1], 'M', [64, 64, 3, 1, 1], [64, 3, 3, 1, 1]]     #decode网络架构图  tran_conv:[inner_channels, outer_channels, filter, stride, padding]
                     #最大逆池化层:M  2*2 步长2

    pool_kernel_size = 2 #池化层核大小
    pool_stride = 2 #池化层步长

    pool_index_list = [] #池化原始所在位置list
    
    def __init__(self, feature_len, img_size):
        """
        初始化函数
        :param: feature_len encode长度
        :param: img_size 图片尺寸           
        """
        
        super(Denoising_AutoEncoder, self).__init__() #等价与nn.Module.__init__()   运用nn.Module初始化
        
        self.img_size = img_size #图片大小
        self.feature_len = feature_len #encode长度
        self.in_channels = 3 #记录输入层数，用于网络输入
        self.en_net = self.encode_conv() #encode网络
        self.en_fc = nn.Linear(in_features = self.in_channels * self.img_size[0] * self.img_size[1], out_features = self.feature_len, bias = True) #encode最后的全连接层       
        self.de_fc = nn.Linear(in_features = self.feature_len, out_features = self.in_channels * self.img_size[0] * self.img_size[1], bias = True) #decode最初的全连接层     
        self.de_net = self.decode_conv() #decode网络     

    def corrupt_x(self, x, cor_rate = 0.01):
        """
        对于x进行随机corrupt
        :param: x 输入变量x
        :param: cor_rate 随机corrupt概率
        return corrupted_x corrupt后的变量x
        """
        judge_matrix = np.random.randn(x.shape) > cor_rate
        corrupted_x = x * judge_matrix
        
        return corrupted_x
    
    def encode_conv(self):
        """
        建立encode部分网络, 不用ReLU稀疏特征, 不用池化层
        return  encode_feature_net  encode部分
        """
        encode_feature_net = []
        for layer_type in self.cfg_conv:
            if layer_type == 'M':
                layer = [nn.MaxPool2d(kernel_size = self.pool_kernel_size, stride = self.pool_stride, return_indices = True)]#最大池化层
                        #对应原始所在位置 
                self.img_size[0] = int((self.img_size[0] - self.pool_kernel_size) / self.pool_stride + 1) #计算图片大小
                self.img_size[1] = int((self.img_size[1] - self.pool_kernel_size) / self.pool_stride + 1)#计算图片大小
            else: #卷积层
                in_channels, out_channels, kernel_size, stride, padding = layer_type
                self.img_size[0] = int((self.img_size[0] - kernel_size + 2 * padding) / stride + 1) #计算图片大小
                self.img_size[1] = int((self.img_size[1] - kernel_size + 2 * padding) / stride + 1)#计算图片大小
                layer = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), #卷积层
                                  #输入通道数     输出特征数    卷积核大小    步长    填充
                         nn.BatchNorm2d(num_features = out_channels), #BN层
                                             #特征数
                         nn.Tanh()]  #Tanh层 
                         #inplace=True改变输入的数据, 节省内存
                self.in_channels = out_channels
                
            encode_feature_net += layer
                  
        return nn.Sequential(*encode_feature_net)
    
    
    def decode_conv(self):
        """
        建立decode部分网络, 不用ReLU稀疏特征, 不用逆池化层
        return  decode_pic_net  encode部分
        """
        decode_pic_net = []
        for layer_type in self.cfg_tranconv:
            if layer_type == 'M':
                layer = [nn.MaxUnpool2d(kernel_size = self.pool_kernel_size, stride = self.pool_stride)]#最大逆池化层
            else: #卷积层
                in_channels, out_channels, kernel_size, stride, padding = layer_type
                layer = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding), #卷积层
                                  #输入通道数     输出特征数    卷积核大小    步长    填充
                         nn.BatchNorm2d(num_features = out_channels), #BN层
                                             #特征数
                         nn.Tanh()]  #Tanh层 
                         #inplace=True改变输入的数据, 节省内存
                #self.in_channels = out_channels
                
            decode_pic_net += layer
        decode_pic_net[-1] = nn.Sigmoid() #最后一层改成Sigmoid 让输入为0-1之间
            
        return nn.Sequential(*decode_pic_net)
    
    
    def forward(self, x):
        """
        前向传播
        :param: x 图片变量
        return code: 图片encode编码 
               decode: encode编码 decode结果
        """
        
        code = self.corrupt_x(x) #corrupt_x
        for net in self.en_net: #encode部分
            if isinstance(net, nn.MaxPool2d):
                code, pool_index = net(code)  #获取池化位置
                self.pool_index_list.append(pool_index)
            else:
                code = net(code)
                
        code = code.view(code.size(0), -1)
        code = self.en_fc(code)
        
        decode = self.de_fc(code)
        decode = decode.view(decode.size(0), self.in_channels, self.img_size[0], self.img_size[1])
        
        for net in self.de_net: #decode部分
            if isinstance(net, nn.MaxUnpool2d):
                decode = net(decode, self.pool_index_list.pop())
            else:
                decode = net(decode)
        
        return code, decode


def get_data(cfg):
    """
    获取预处理和数据增强后的数据集
    :param: cfg 配置文件
    return trainloader, testloader, data_ok
           训练loader   测试loader  数据是否获得
    """
    print('>' * 80)
    print('Date getting begin')
    print('')
    
    try: 
        #训练集数据获取
        
        #训练集数据预处理
        trainloader, testloader = data_get(cfg)
       
        print('Succeeded to get_data')
        print('>' * 80)
        print(' ')
        print(' ')
        
        return trainloader, testloader, True 
    
    except: #防爆
        
        print('Failed to get_data, stop training')
        print('>' * 80)
        print(' ')
        print(' ')
        
        return None, None, False
    
"""
通用模型初始化
"""
def initialize_weights(model):
    """
    模型初始化
    :param:model 输入模型 可以用model.apply(initialize_weights)调用
    """
    for module in model.modules(): #模型中的所有模式，包含总，序列，层
        
        if isinstance(module, nn.Conv2d): #卷积层
            n_conv = module.kernel_size[0] * module.kernel_size[1] * module.out_channels #卷积权重元素个数
            module.weight.data.normal_(0, math.sqrt(2. / n_conv)) #卷积核初始化 正态随机数, 限制标准差
            if module.bias is not None: #有偏移项
                module.bias.data.zero_() #偏移项初始化 = 0
                
        elif isinstance(module, nn.BatchNorm2d): #BN层
            module.weight.data.fill_(1) #归一化权重 == 标准差, 初始化 = 1
            module.bias.data.zero_() #归一化偏移项 == 均值, 初始化 = 0
            
        elif isinstance(module, nn.Linear): #全连接层
            n_fc = module.in_features * module.out_features #全连接层权重个数
            module.weight.data.normal_(0, math.sqrt(2. / n_fc)) #全连接权重正态初始化
            module.bias.data.zero_()#偏移项初始化 = 0
            
        elif isinstance(module, nn.ConvTranspose2d): #反卷积层
            n_conv = module.kernel_size[0] * module.kernel_size[1] * module.out_channels #卷积权重元素个数
            module.weight.data.normal_(0, math.sqrt(2. / n_conv)) #卷积核初始化 正态随机数, 限制标准差
            if module.bias is not None: #有偏移项
                module.bias.data.zero_() #偏移项初始化 = 0
                
#模型训练
def train(trainloader, cfg):
    """
    :param: trainloader 训练数据loader
    :param: cfg 配置文件
    return train_loss 各epoch训练损失函数list
    """
    print('>' * 80)    
    print('Begin train')
    print(' ')

    #模型基本配置
    print('Model use {}'.format(cfg.model_name))
    print(' ')
    
    model = Denoising_AutoEncoder(cfg.classes_num, list(cfg.resize_size)) #载入模型

    begin_epoch = 0 #初始epoch
    if cfg.pretrain: #如果有预训练
        model_info = torch.load(os.path.join(cfg.pretrain, 'model_{}_state.pkl'.format(cfg.model_name)))
        model.load_state_dict(model_info['state_dict']) #加载训练出的模型
        begin_epoch = model_info['epoch'] + 1

    model.train() #切换到训练模式
          
    device = torch.device(cfg.device) #选择设备
    try: #有gpu
        model.cuda(device) #设备选择
        gpu = True #是否有gpu
        print('Gpu is used')
    except:#不用gpu
        gpu = False
        print('Cpu is used')
    
    criterion = torch.nn.MSELoss() #损失函数方法：MSE
    alpha = cfg.alpha if not cfg.pretrain else model_info['alpha'] #初始学习率，有预处理以预处理为准
    optimizer = torch.optim.SGD(model.parameters(), momentum = cfg.momentum, lr = alpha, weight_decay = cfg.weight_decay) #迭代方法SGD
                                                           #动量              初始学习率            权重衰减趋势
    #optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas = (0.9, 0.999), eps = 1e-8, weight_decay = cfg.weight_decay) #迭代方法Adam
                                                       #学习率      梯度及梯度平方系数   分母防零修正           权重衰减系数
                                                       
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10) #学习率衰减(余弦退火)
                                                        #0, T_max下降，T_max到2 * T_max上升
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda step:np.sin(step) / step) #自己设定,函数输入为步数
                                                                    #自己设定函数
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [20, 80], gamma = 0.9) #分段式衰减
                                                                #设定变化点，遇到该点变化  衰减系数
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99) #指数衰减，每个epoch
                                                                   #衰减系数
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 10, verbose = False, threshold = 0.0001, threshold_mode = 'rel', cooldown = 0, min_lr = 0, eps = 1e-08) #自适应
                                                                     #检测loss减小     衰减系数       容忍次数        是否print         变化阈值范围        rel比例 abs值           冷却时间      最小lr     效果较差不变                                  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = cfg.lr_step, gamma = cfg.lr_decay) #学习率线性衰减
                                                                 #衰减步长         衰减系数 lr *= lr_decay
    if not cfg.pretrain: #如果没预训练
        model.apply(initialize_weights) #模型初始化，内置初始化，均匀分布
    
    if gpu:
        model, optimizer = amp.initialize(model, optimizer, opt_level = "O1", verbosity = 0) #混合精度模型
                                              #Oo fp32, O1混合, O2几乎fp16, O3 fp16                        
    if cfg.pretrain: #如果预训练
        train_loss = model_info['train_loss']
    else:
        train_loss = []
    
    total_start_time = time.time() #记录时间
    start_time = time.time() #记录时间
    
    for epoch in tqdm(range(begin_epoch, cfg.epoch_num)): #迭代全图
    #for epoch in range(1, cfg.epoch_num + 1): #迭代全图  
        
        train_loss_i = 0 #第i次epoch损失
        for batch_idx, (imgs, _) in enumerate(trainloader): #迭代批次
            #批数       图片
            
            #one-hot label 化：(交叉熵里面自动有)
            #classes = torch.zeros(cfg.batch_size, len(cfg.classes)).scatter_(1, classes.view(len(classes),1), 1)
                                                                     #稀疏化 维度       值                  对应标签值           
            if gpu: #用gpu
                imgs = imgs.cuda(device) #将数据移到GPU上
                inputs= Variable(imgs)  #变量化输入x,y
         
            optimizer.zero_grad()   # 先将optimizer梯度先置为0
            
            encode1, decode1 = model(inputs[:,:,0,:,:]) #前向传播
            encode2, decode2 = model(inputs[:,:,1,:,:]) #前向传播

            #outputs = model.forward(inputs) #等价效果
            
            loss = criterion(inputs[:,:,0,:,:], decode1) + criterion(inputs[:,:,1,:,:], decode2) - criterion(decode1, decode2)
            #损失函数

            if gpu:
                with amp.scale_loss(loss, optimizer) as scaled_loss:#采用混合精度模型         
                    scaled_loss.backward() 
            else:
                loss.backward()  #反向传播，计算梯度
            
            optimizer.step() #更新参数
        
            train_loss_i += loss.data.item()#记录每次训练Loss, 必须loss.data[0]
            
        
        scheduler.step() #学习率记录step      
        train_loss.append(train_loss_i) #记录每轮的损失函数值
        
        if  epoch % 10 == 9: #每十次迭代            
            end_time = time.time() #记录时间
            
            #展示模型训练状态
            print(' ')
            print('>' * 80)    
            print('Epoch : {} - {}'.format(epoch - 8, epoch + 1))
            print('Training_time = {} s / epoch'.format(str( (end_time - start_time) / 10 )[:8]) )
            print('Avg_loss_function = {}'.format(np.mean(train_loss[-10:])))
            print('>' * 80)
            print(' ')
            
            if not cfg.mix_up and epoch / cfg.epoch_num > 0.1: #预热10%迭代
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 
                            'train_loss': train_loss, 'alpha': optimizer.state_dict()['param_groups'][0]['lr']}, #记录迭代次数，状态字典，最好结果, 损失函数list, 学习率
                            os.path.join(cfg.save_path, 'model_{}_state.pkl'.format(cfg.model_name))) #最好的结果(覆盖原来的)
                
                #torch.save(model.state_dict(), os.path.join(cfg.save_path, 'model_{}_state.pkl'.format(cfg.model_name))) #最好的结果(覆盖原来的)
                #保存中间最好的模型(以后可以再训练) 保存模型所有信息，读取时要载入框架
                #torch.save(model, './model_{}.pkl'.format(cfg.model_name)) #保存模型信息，读取时直接读取 等价

            start_time = time.time() #更新时间
            
        torch.cuda.empty_cache() #清理显存
         
    total_end_time = time.time() #记录时间           
    
    print('Training time = {} s / epoch'.format( str( (total_end_time - total_start_time) / cfg.epoch_num )[:8] ) )
    print(' ')
    print('Finish train')
    print('>' * 80)    
    print(' ')
    print(' ')
    
    return train_loss


    
#模型测试
def test(testloader, cfg):
    """
    测试Auto_encoder效果
    :param: testloader 测试数据loader
    :param: cfg 配置文件
    return: precision 准确率
    """
    print('>' * 80)
    print('Begin test')
    print(' ')
    
    if not os.path.exists(os.path.join(cfg.save_path, 'ae_result')):
        os.mkdir(os.path.join(cfg.save_path, 'ae_result'))
        
    #载入模型结构   
    model = Denoising_AutoEncoder(cfg.classes_num, list(cfg.resize_size)) #载入模型
    
    model_info = torch.load(os.path.join(cfg.save_path, 'model_{}_state.pkl'.format(cfg.model_name))) #获取字典
    model.load_state_dict(model_info['state_dict']) #加载训练出的模型
    
    #model.load_state_dict(torch.load(os.path.join(cfg.save_path, 'model_{}_state.pkl'.format(cfg.model_name)))) #加载训练出的模型
    #model = torch.load(r'./model_{}.pkl'.format(cfg.model_name)) 等价
    
    model.eval() #测试，不改变权重

    device = torch.device(cfg.device) #选择设备
    try: #用gpu
        model.cuda(device) #设备选择
        gpu = True #是否有gpu
        print('Gpu is used')
    except:#不用gpu
        gpu = False
        print('Cpu is used')
    
    total = 0 #图片总数

    with torch.no_grad(): #不进行反向传播, 减少内存
    
        start_time = time.time() #记录时间
        
        for idx, (imgs, _) in enumerate(testloader): #遍历图片
           #索引  图片  
           if gpu:#用gpu
               imgs= imgs.cuda()   # 将数据移到GPU上
           inputs = Variable(imgs) #变量化输入x
           
           encode, decode = model(inputs) #运行模型(获得结果)
           
           dec_pic = np.uint8(decode.cpu().numpy()[0, :, :, :] * 255) #*255 + 转格式
           dec_pic = np.transpose(dec_pic, (1,2,0)) #调整通道位置
           dec_pic_gray = cv2.cvtColor(dec_pic, cv2.COLOR_RGB2GRAY) #灰度图
           cv2.imwrite(os.path.join(cfg.save_path, 'ae_result', str(idx) + '.png'), dec_pic_gray) #储存图片
           
           #dec_pic = transforms.ToPILImage()(decode.cpu()[0, :, :, :])       
           #dec_pic.save(os.path.join(cfg.save_path, 'ae_result', str(idx) + '.png')) #等价方法
           
           total += 1
        
        end_time = time.time()
             
        torch.cuda.empty_cache() #清理显存
    
    print('Decoding time = {} s / pic'.format( str( (end_time - start_time) / total )[:8] ) )
    print(' ')
    print('Finish test')
    print('>' * 80)    
    print(' ')
    print(' ')
        
    
    
def main(config_path = r'./config_ae.yml'):
    """
    主函数, 完成数据载入和模型训练 + 测试
    :param: config_path config路径
    return: train_loss,        precision,       wronglist       
            各epcoh下训练损失   测试准确性       错判断序号列表
    """
    cfg = Config(config_path) #获取配置文件
    trainloader, testloader, data_ok = get_data(cfg) #获取数据并进行数据预处理和增强
    if data_ok:
        train_loss = train(trainloader, cfg) #模型训练
        test(testloader, cfg) #模型训练
        return train_loss
    
    else:
        return None


if __name__ == "__main__":        
    train_loss = main(r'./config_ae.yml')
    while True:
        pass


