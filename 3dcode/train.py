from __future__ import print_function
import sys
sys.path.append(r'/home/sufedc_nvidia_newgyh/apex')
from apex import amp
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
warnings.filterwarnings("ignore")
from config import Config #获取配置
from Pytorch_nets_channel1 import * #自定义的神经网络
from Data_Augment_New import * #自定义的数据增强方法 对PIL Image
from CV_tricks import * #自定义的CV的tricks
from Data_loader2d import  data_get

def train(model, trainloader, cfg):
    """
    :param: trainloader 训练数据loader
    :param: cfg 配置文件
    return train_loss 各epoch训练损失函数list
    """
    print('>' * 80)    
    print('Begin train')
    print(' ')

    #模型基本配置
    begin_epoch = 0 #初始epoch
    model.train() #切换到训练模式
    # if not cfg.distribution:
    #     #device = torch.device(cfg.device) #选择设备
    #     try: #有gpu
    #         #model.cuda(device) #设备选择
    #         model.cuda()
    #         gpu = True #是否有gpu
    #         print('Gpu is used')
    #     except:#不用gpu
    #         gpu = False
    #         print('Cpu is used')
    # else:
    #     gpu = True #是否有gpu
    #     print('Gpu is used')
    # if cfg.pretrain: #如果有预训练
    #     model_info = torch.load(os.path.join(cfg.pretrain, 'model_{}_state.pkl'.format(cfg.model_name)))

    criterion = torch.nn.CrossEntropyLoss() #损失函数方法：交叉熵（自带softmax）
    alpha = cfg.alpha                       #初始学习率，有预处理以预处理为准
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
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = cfg.lr_decay, patience = 5, verbose = False, threshold = 0.0001, threshold_mode = 'rel', cooldown = 0, min_lr = 0, eps = 1e-08) #自适应
                                                                     #检测loss减小            衰减系数       容忍次数        是否print         变化阈值范围        rel比例 abs值           冷却时间      最小lr  效果较差不变                                  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = cfg.lr_step, gamma = cfg.lr_decay) #学习率线性衰减
                                                                #衰减步长         衰减系数 lr *= lr_decay

    
    if gpu:
        model, optimizer = amp.initialize(model, optimizer, opt_level = "O1", verbosity = 0) #混合精度模型
                                              #Oo fp32, O1混合, O2几乎fp16, O3 fp16   
    if cfg.distribution:
        model = torch.nn.DataParallel(model, device_ids = list(cfg.distribution))

    # if cfg.pretrain: #如果有预训练
    #     model_info = torch.load(os.path.join(cfg.pretrain, 'model_{}_state.pkl'.format(cfg.model_name)))
    #     model.load_state_dict(model_info['state_dict']) #加载训练出的模型
    #     begin_epoch = model_info['epoch'] + 1        
    #     if not cfg.mix_up:
    #         best_correct = model_info['best_correct']
    #     train_loss = model_info['train_loss']
    # else:
    #     best_correct = 0 #最优的正确数
    #     train_loss = []
    
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

                # if gpu: #用gpu
                #     imgs, classes_a, classes_b = imgs.cuda(), classes_a.cuda(), classes_b.cuda() #将数据移到GPU上          
                # inputs, targets_a, targets_b = Variable(imgs), Variable(classes_a),  Variable(classes_b) #变量化输入x,y_a,y_b
                """
                imgs, classes = mix_up_onehot_data(imgs, classes) #one-hot label数据mix-up
                if gpu: #用gpu
                    imgs, classes = imgs.cuda(), classes.cuda() #将数据移到GPU上
                inputs, targets = Variable(imgs), Variable(classes)  #变量化输入x,y
                """
            # else: #没用mix_up策略
            #     if gpu: #用gpu
            #         imgs, classes = imgs.cuda(), classes.cuda() #将数据移到GPU上
            #     inputs, targets = Variable(imgs), Variable(classes)  #变量化输入x,y
         
            optimizer.zero_grad()   # 先将optimizer梯度先置为0            
            outputs = model(inputs) #前向传播
            #outputs = model.forward(inputs) #等价效果
            if cfg.mix_up: #如果mix_up
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b) # 计算mix-up损失函数
            else: #如果没mix_up
                loss = criterion(outputs, targets) #损失函数
                
            #loss = criterion(outputs, targets)
            
            if 1:
                with amp.scale_loss(loss, optimizer) as scaled_loss:#采用混合精度模型         
                    scaled_loss.backward() 
            else:
                loss.backward()  #反向传播，计算梯度
            #loss.backward()  #反向传播，计算梯度
            optimizer.step() #更新参数
            train_loss_i += loss.data.item()#记录每次训练Loss, 必须loss.data[0]
            if not cfg.mix_up: #没有mix_up下才有中间结果
                _, predicted = torch.max(outputs.data, dim = 1) #获得预测结果，结果为批次数据, 所以行最大(一行一个结果)
                correct += predicted.eq(targets.data).cpu().sum().item() #计算正确的图片数，cpu上算,.tensor.item()获取值
                
            total += inputs.size(0)#图片数加总(size第一维为批大小), size为大小
        
        scheduler.step() #学习率记录step      
        train_loss.append(train_loss_i) #记录每轮的损失函数值
        precision = 100. * correct / total #准确率 
    
        if  epoch % 5 == 4: #每五次迭代            
            end_time = time.time() #记录时间
            #展示模型训练状态
            print(' ')
            print('>' * 80)    
            print('Epoch : {} - {}'.format(epoch - 3, epoch + 1))
            print('Training_time = {} s / epoch'.format(str( (end_time - start_time) / 5 )[:8]) )
            print('Avg_loss_function = {}'.format(np.mean(train_loss[-5:])))
            if not cfg.mix_up:
                print('Precision = {} %'.format(precision))            
            print('>' * 80)
            print(' ')
            
            if not cfg.mix_up and epoch / cfg.epoch_num > 0.1 and correct >= best_correct: #预热10%迭代, 更好的模型，mix_up下没法比较
                best_correct = correct #更新最优结果
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'best_correct': best_correct,
                            'train_loss': train_loss, 'alpha': optimizer.state_dict()['param_groups'][0]['lr']}, #记录迭代次数，状态字典，最好结果, 损失函数list, 学习率
                            os.path.join(cfg.save_path, 'model_{}_state.pkl'.format(cfg.model_name))) #最好的结果(覆盖原来的)
            
            if cfg.mix_up:
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                            'train_loss': train_loss, 'alpha': optimizer.state_dict()['param_groups'][0]['lr']}, #记录迭代次数，状态字典，最好结果, 损失函数list, 学习率
                            os.path.join(cfg.save_path, 'model_{}_state.pkl'.format(cfg.model_name)))
                
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
    return model, train_loss
