# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 20:09:47 2020

@author: 12057
"""

from __future__ import print_function
from apex import amp

import torch
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import PIL
import os
import time
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from config import Config #获取配置
from Pytorch_nets import * #自定义的神经网络
from Data_Augment_New import * #自定义的数据增强方法 对PIL Image
from CV_tricks import * #自定义的CV的tricks

"""
epoch: 全部图片迭代一次
iteration: 一个batch迭代一次
"""


#获取数据，并进行预处理和数据增强

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
        #训练集预处理
        transform_train = transforms.Compose(
            [#自定义数据增强方法
            #transforms.Lambda(lambda img: sidecrop(img)), #必须裁边的图像裁剪
            transforms.Lambda(lambda img: addblack(img)) #图像加黑边
            ] + \
            [#传统数据增强
            #transforms.RandomCrop(size = cfg.cut_size, padding = 4), #随机裁剪 + 缩放
            transforms.RandomHorizontalFlip(p = 0.5), #随机水平翻转
            transforms.RandomResizedCrop(size = cfg.resize_size[0], scale = cfg.cut_scale, ratio = cfg.cut_ratio), #随机长宽比范围裁剪,再缩放
                                                 #缩放后大小,整数    面积比例范围        长宽比比例范围
            transforms.Resize(size = cfg.resize_size, interpolation = PIL.Image.BILINEAR), #调整大小一致，所有图片大小需要一致
                                                              #插值方法
            transforms.ToTensor(), #tensor化：固定操作  函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
            transforms.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)) #标准化：固定操作
            ])
    
        #测试集预处理
        transform_test = transforms.Compose([
            transforms.Resize(size = cfg.resize_size, interpolation = PIL.Image.BILINEAR), #调整大小一致，所有图片大小需要一致
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)),
        ])
        
        if cfg.data_info == 'CIFAR10':
            #CIFAR10数据   
            #训练集数据获取
            trainset = torchvision.datasets.CIFAR10(root = r'./CIFAR10_data', train = True, download = True, transform = transform_train)
                                                         #目标路径              是否训练     是否下载,已下载也没事      数据变换类型  
            #测试集数据获取
            testset = torchvision.datasets.CIFAR10(root = r'./CIFAR10_data', train = False, download = True, transform = transform_test)
        
        else:
            #自定义数据   
            #训练集数据获取
            trainset = torchvision.datasets.ImageFolder(os.path.join(cfg.data_info, 'train'), transform = transform_train)
                                                            #数据集所在文件夹，一个文件夹为一类        
            #测试集数据获取
            testset = torchvision.datasets.ImageFolder(os.path.join(cfg.data_info, 'test'), transform = transform_test)
            
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = cfg.batch_size, shuffle = True, num_workers = 0)
                                                  #数据集   批大小(自动化)               是否每次打乱(一般True)  多进程数
        testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 0)
                                                         #测试集不需要batch和shuffle   
            
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
    model = get_model(cfg.model_name, cfg.classes) #根据模型名载入模型

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
    
    criterion = torch.nn.CrossEntropyLoss() #损失函数方法：交叉熵（自带softmax）
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
    if not cfg.pretrain: #如果预训练
        model.apply(initialize_weights) #模型初始化，内置初始化，均匀分布
    
    if gpu:
        model, optimizer = amp.initialize(model, optimizer, opt_level = "O1", verbosity = 0) #混合精度模型
                                              #Oo fp32, O1混合, O2几乎fp16, O3 fp16                        
    if cfg.pretrain: #如果预训练
        best_correct = model_info['best_correct']
        train_loss = model_info['train_loss']
    else:
        best_correct = 0 #最优的正确数
        train_loss = []
    
    total_start_time = time.time() #记录时间
    start_time = time.time() #记录时间
    
    for epoch in tqdm(range(begin_epoch, cfg.epoch_num)): #迭代全图
    #for epoch in range(1, cfg.epoch_num + 1): #迭代全图  
        correct = 0 #正确的图片数量  
        total = 0 #图片总数
        
        train_loss_i = 0 #第i次epoch损失
        for batch_idx, (imgs, classes) in enumerate(trainloader): #迭代批次
            #批数       图片    类别 
            
            #one-hot label 化：(交叉熵里面自动有)
            #classes = torch.zeros(cfg.batch_size, len(cfg.classes)).scatter_(1, classes.view(len(classes),1), 1)
                                                                     #稀疏化 维度       值                  对应标签值           
            if cfg.mix_up:  #mix_up策略
                imgs, classes_a, classes_b, lam = mix_up_data(imgs, classes)  #非one-hot label数据mix-up
                if gpu: #用gpu
                    imgs, classes_a, classes_b = imgs.cuda(), classes_a.cuda(), classes_b.cuda() #将数据移到GPU上          
                inputs, targets_a, targets_b = Variable(imgs), Variable(classes_a),  Variable(classes_b) #变量化输入x,y_a,y_b
            else: #没用mix_up策略
                if gpu: #用gpu
                    imgs, classes = imgs.cuda(), classes.cuda() #将数据移到GPU上
                inputs, targets = Variable(imgs), Variable(classes)  #变量化输入x,y
         
            optimizer.zero_grad()   # 先将optimizer梯度先置为0
            
            outputs = model(inputs) #前向传播
            #outputs = model.forward(inputs) #等价效果
            
            if cfg.mix_up: #如果mix_up
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b) # 计算mix-up损失函数
            else: #如果没mix_up
                loss = criterion(outputs, targets) #损失函数

            if gpu:
                with amp.scale_loss(loss, optimizer) as scaled_loss:#采用混合精度模型         
                    scaled_loss.backward() 
            else:
                loss.backward()  #反向传播，计算梯度
            
            optimizer.step() #更新参数
        
            train_loss_i += loss.data.item()#记录每次训练Loss, 必须loss.data[0]
            
            if not cfg.mix_up: #没有mix_up下才有中间结果
                _, predicted = torch.max(outputs.data, dim = 1) #获得预测结果，结果为批次数据, 所以行最大(一行一个结果)
                correct += predicted.eq(targets.data).cpu().sum().item() #计算正确的图片数，cpu上算,.tensor.item()获取值
                total += targets.size(0)#图片数加总(size第一维为批大小), size为大小
        
        scheduler.step() #学习率记录step      
        train_loss.append(train_loss_i) #记录每轮的损失函数值
        precision = 100. * correct / total #准确率 
        
        if  epoch % 10 == 9: #每十次迭代            
            end_time = time.time() #记录时间
            
            #展示模型训练状态
            print(' ')
            print('>' * 80)    
            print('Epoch : {} - {}'.format(epoch - 8, epoch + 1))
            print('Training_time = {} s / epoch'.format(str( (end_time - start_time) / 10 )[:8]) )
            print('Avg_loss_function = {}'.format(np.mean(train_loss[-10:])))
            if not cfg.mix_up:
                print('Precision = {} %'.format(precision))            
            print('>' * 80)
            print(' ')
            
            if not cfg.mix_up and epoch / cfg.epoch_num > 0.1 and correct > best_correct: #预热10%迭代, 更好的模型，mix_up下没法比较
                best_correct = correct #更新最优结果
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'best_correct': best_correct,
                            'train_loss': train_loss, 'alpha': optimizer.state_dict()['param_groups'][0]['lr']}, #记录迭代次数，状态字典，最好结果, 损失函数list, 学习率
                            os.path.join(cfg.save_path, 'model_{}_state.pkl'.format(cfg.model_name))) #最好的结果(覆盖原来的)
                
                #torch.save(model.state_dict(), os.path.join(cfg.save_path, 'model_{}_state.pkl'.format(cfg.model_name))) #最好的结果(覆盖原来的)
                #保存中间最好的模型(以后可以再训练) 保存模型所有信息，读取时要载入框架
                #torch.save(model, './model_{}.pkl'.format(cfg.model_name)) #保存模型信息，读取时直接读取 等价

            start_time = time.time() #更新时间
            
        torch.cuda.empty_cache() #清理显存
        
    if not os.path.exists(os.path.join(cfg.save_path, 'model_{}_state.pkl'.format(cfg.model_name))): #mix_up下 或者没更好结果 下保存最后结果
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'best_correct': best_correct, 
                    'train_loss': train_loss, 'alpha': optimizer.state_dict()['param_groups'][0]['lr']}, #记录迭代次数，状态字典，最好结果, 损失函数list, 学习率
                    os.path.join(cfg.save_path, 'model_{}_state.pkl'.format(cfg.model_name))) #最好的结果(覆盖原来的)
 
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
    :param: testloader 测试数据loader
    :param: cfg 配置文件
    return: precision 准确率
    """
    print('>' * 80)
    print('Begin test')
    print(' ')

    #载入模型结构   
    model = get_model(cfg.model_name, cfg.classes) #根据模型名载入模型
    
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
    
    correct = 0 #正确数
    total = 0 #图片总数
    wronglist = [] #分错序号
    with torch.no_grad(): #不进行反向传播, 减少内存
        
        start_time = time.time() #记录时间
        
        for idx, (imgs, classes) in enumerate(testloader): #遍历图片
           #索引  图片   类别
           if gpu:#用gpu
               imgs, classes = imgs.cuda(), classes.cuda()   # 将数据移到GPU上
           inputs, targets = Variable(imgs), Variable(classes) #变量化输入x,y
           
           outputs = model(inputs) #运行模型(获得结果)
           
           _, predicted = torch.max(outputs.data, dim = 1) #获得预测结果，结果为批次数据, 所以行最大(一行一个结果)
           correct += predicted.eq(targets.data).cpu().sum().item() #正确数,cpu上算
           
           if predicted.eq(targets.data).cpu().sum().item() != 1: #如果判断错
               wronglist.append((idx, classes.item()))
               
           total += targets.size(0) #图片数加总(size第一维为批大小), size为大小

        end_time = time.time()
     
        precision = 100. * correct / total #准确率
        
        torch.cuda.empty_cache() #清理显存
    
    print('Infering time = {} s / pic'.format( str( (end_time - start_time) / total )[:8] ) )
    print(' ')
    print('Finish test')
    print('>' * 80)    
    print(' ')
    print(' ')
    
    print('Test Precision = {} %'.format(precision))   
    
    return precision, wronglist


def main(config_path):
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
        precision, wronglist = test(testloader, cfg) #模型测试
    
        return train_loss, precision, wronglist 
    
    else:
        return None, None, None


if __name__ == "__main__":        
    train_loss, precision, wronglist = main(r'./config.yml')
    while True:
        pass
    