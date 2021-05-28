# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:39:01 2020

@author: 12057
"""

import math

import torch.nn as nn #类
#import torch.nn.functional as F #函数
#层内有variable的情况用nn定义，否则用nn.functional定义

"""
本质上nn.是一个子网络,为类 +()构建实例, F为函数(参数)
nn有大写, F一般小写
Dropout 进入该层会部分失活, 一般在fc前, nn.Dropout() > F.dropout() eval时前者会关闭  
"""

"""
Pytorch 数据大小: batch_size * channels * Height * Width
same-padding要手算
"""

"""
定义网络可以不设置_make_layers, _classifer_layers以在forward中获取中间过程
nn.Sequential为了结构清楚 + 封装, 类似列表
"""

"""
分类前fc层改为GAP，输入图片可以任意大小
"""


"""
通常的CNN通用格式，修改cfg就行, VGGs为简洁不这样做
3d为 batchsize, channels, D(深度), H, W
"""
#LeNet神经网络
class LeNet(nn.Module): #nn.Module为父类
    """
    LeNet神经网络结构搭建, 分类前fc层改为GAP, 加入BN, 激活函数为ReLU, 欠拟合严重，加入NiN增加非线性
    """
    cfg = [[1, 6, 5, 1, 2], 'M', [6, 16, 5, 1, 0], 'M', [16, 16, 5, 1, 0], [16, 120, 1, 1, 0], [120, 120, 1, 1, 0]]   
    #LeNet网络架构图  conv:[inner_channels, outer_channels, filter, stride, padding]
                      #最大池化层:M  2*2 步长2
    pool_kernel_size = 2 #池化层核大小
    pool_stride = 2 #池化层步长
    
    def __init__(self, class_num): #选择目标分类数
        """
        网络初始化
        param: class_num  目标分类数
        """
        super(LeNet, self).__init__()   # 等价与nn.Module.__init__()   运用nn.Module初始化
        #super为调用父类初始化
        self.class_num = class_num #目标分类数
        
        self.features = self._make_layers()  #搭建LeNet网络特征提取部分网络
        self.classifier = self._classifer_layers() #搭建LeNet网络分类部分网络 
        self.in_channels = 1 #记录输入层数，用于分类层输入
        
    def _make_layers(self):
        """
        建立特征提取部分网络
        return  LeNet_feature_net  LeNet特征提取部分
        """
        layers = []
        for layer_type in self.cfg:
            if layer_type == 'M':
                layer = [nn.MaxPool3d(kernel_size = (1, self.pool_kernel_size, self.pool_kernel_size), stride = (1, self.pool_stride, self.pool_stride) )] #最大池化层
                                                        #池化核大小  第一个核为深度对应的，所以取1                                    步长
            else: #卷积层
                in_channels, out_channels, kernel_size, stride, padding = layer_type
                layer = [nn.Conv3d(in_channels, out_channels, (1, kernel_size, kernel_size), (1, stride, stride), (0, padding, padding)), #卷积层
                                  #输入通道数     输出特征数              卷积核大小                 步长            填充
                         nn.BatchNorm3d(num_features = out_channels), #BN层
                                             #特征数
                         nn.ReLU(inplace = True)]  #ReLU层 
                         #inplace=True改变输入的数据, 节省内存
                self.in_channels = out_channels

            layers += layer
            
        layers.append(nn.AdaptiveAvgPool3d(output_size = (1, 1, 1))) #全局平均池化层，化简特征 -> 120 * 1 * 1, robust，且接受任意大小图片
                                               #输出大小
        LeNet_feature_net = nn.Sequential(*layers) #序列化
        
        return LeNet_feature_net
       
    def _classifer_layers(self):
        """
        建立分类部分网络
        return LeNet_classifer_net LeNet分类部分
        """
        
        layers = []
        #layers.append(nn.Dropout(p = 0.2)) #dropout层 
                                   #概率值
        layers.append(nn.Linear(in_features = self.in_channels, out_features = self.class_num, bias = True)) #分类层(最后的全连接层)
                                   #输入特征数               输出特征数           偏移项 默认True
        #layers.append(nn.Softmax(dim = 0)) #softmax层，交叉熵里面包含，分类不需要softmax, 且对于非one-hot label多分类不适用
                             #dim = 0列求softmax        
        LeNet_classifer_net = nn.Sequential(*layers) #序列化（等价于列表)
        
        return LeNet_classifer_net
            
    def forward(self, x):
        """
        前向传播
        :param: x 图片变量
        return out: 前向传播输出
        """
        if x.size(1) != 1: #如果不是单通道图片
            R, G, B = x[:,0,:,:,:], x[:,1,:,:,:], x[:,2,:,:,:] # R,G,B图像 对PIL图
            x = R * 0.299 + G * 0.587 + B * 0.114 #灰度图公式，加权平均
            x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3)) #保留原始格式 batch_size * channels * D * H * W
            
            #x = torch.mean(x, dim = 1, keepdim = True)  #按照通道数简单平均, 修正为单通道，效果不好
                            #平均的维度   是否保留原来维度，填1
            
        feature = self.features(x) #获取x对应特征
        flat_feature = feature.view(feature.size(0), -1) #view类似resize，特征化为一维
        out = self.classifier(flat_feature) #获取x分类结果

        return out



#AlexNet神经网络
class AlexNet(nn.Module): #nn.Module为父类
    """
    AlexNet神经网络结构搭建(LRN换为BN, 前几层fc换为全局池化)
    """
    cfg = [[3, 96, 11, 4, 0], 'M', [96, 256, 5, 1, 2], 'M',
           [256, 384, 3, 1, 1], [384, 384, 3, 1, 1], [384, 256, 3, 1, 1], 'M']
    #AlexNet网络架构图  conv:[inner_channels, outer_channels, filter, stride, padding]
                      #最大池化层:M  3*3 步长2
    pool_kernel_size = 3 #池化层核大小
    pool_stride = 2 #池化层步长
    
    def __init__(self, class_num): #选择目标分类数
        """
        网络初始化
        param: class_num  目标分类数
        """
        super(AlexNet, self).__init__()   # 等价与nn.Module.__init__()   运用nn.Module初始化
        #super为调用父类初始化
        self.class_num = class_num #目标分类数
        
        self.in_channels = 3 #记录输入层数，用于分类层输入

        self.features = self._make_layers()  #搭建AlexNet网络特征提取部分网络
        self.classifier = self._classifer_layers() #搭建AlexNet网络分类部分网络 
                
    def _make_layers(self):
        """
        建立特征提取部分网络
        return  AlexNet_feature_net  AlexNet特征提取部分
        """
        layers = []
        for layer_type in self.cfg:
            if layer_type == 'M':
                layer = [nn.MaxPool3d(kernel_size = (1, self.pool_kernel_size, self.pool_kernel_size), stride = (1, self.pool_stride, self.pool_stride) )] #最大池化层
                                                                    #池化核大小                                       步长
                in_channels, out_channels, kernel_size, stride, padding = layer_type
                layer = [nn.Conv3d(in_channels, out_channels, (1, kernel_size, kernel_size), (1, stride, stride), (0, padding, padding)), #卷积层
                                  #输入通道数     输出特征数              卷积核大小                 步长            填充
                         nn.BatchNorm3d(num_features = out_channels), #BN层
                                             #特征数
                         nn.ReLU(inplace = True)]  #ReLU层 
                         #inplace=True改变输入的数据, 节省内存
                self.in_channels = out_channels
            layers += layer
            
        layers.append(nn.AdaptiveAvgPool3d(output_size = (1, 1, 1))) #全局平均池化层，化简特征 -> 256 * 1 * 1, robust，且接受任意大小图片
                                               #输出大小
        AlexNet_feature_net = nn.Sequential(*layers) #序列化
        
        return AlexNet_feature_net
       
    def _classifer_layers(self):
        """
        建立分类部分网络
        return AlexNet_classifer_net AlexNet分类部分
        """
        
        layers = []
        layers.append(nn.Dropout(p = 0.2)) #dropout层
                                   #概率值
        layers.append(nn.Linear(in_features = self.in_channels, out_features = self.class_num, bias = True)) #分类层(最后的全连接层)
                                   #输入特征数               输出特征数           偏移项 默认True
        #layers.append(nn.Softmax(dim = 0)) #softmax层，交叉熵里面包含，分类不需要softmax, 且对于非one-hot label多分类不适用
                             #dim = 0列求softmax        
        AlexNet_classifer_net = nn.Sequential(*layers) #序列化（等价于列表)
        
        return AlexNet_classifer_net
            
    def forward(self, x):
        """
        前向传播
        :param: x 图片变量
        return out: 前向传播输出
        """
        feature = self.features(x) #获取x对应特征
        flat_feature = feature.view(feature.size(0), -1) #view类似resize，特征化为一维
        out = self.classifier(flat_feature) #获取x分类结果
    
        return out



#VGG类神经网络
class VGGs(nn.Module): #nn.Module为父类
    """
    VGGs神经网络结构搭建, 前几层fc换为全局池化, 尝试深度可分离卷积
    """
    cfg = {
            'VGG_11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG_13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG_16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG_19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
            } #不同VGG网络架构图   CONV：输出通道数   最大池化层:M 2*2 步长2
                                 #padding为same
    pool_kernel_size = 2 #池化层核大小
    pool_stride = 2 #池化层步长
    
    def __init__(self, vgg_name, class_num): #选择不同VGG网络,以及目标分类数
        """
        网络初始化
        param: vgg_name 选择的VGG网络名
        param: class_num  目标分类数
        """
        super(VGGs, self).__init__()   # 等价与nn.Module.__init__()   运用nn.Module初始化
        #super为调用父类初始化
        self.class_num = class_num #目标分类数
        self.vgg_name = vgg_name #选择VGG模型名字
        
        self.in_channels = 3 #记录输入层数，用于分类层输入

        self.features = self._make_layers()  #搭建vgg_name对应vgg网络特征提取部分网络
        self.classifier = self._classifer_layers() #搭建vgg网络分类部分网络
     
    def _make_layers(self):
        """
        建立特征提取部分网络
        return  vgg_feature_net VGG特征提取部分
        """
        layers = []
        for layer_type in self.cfg[self.vgg_name]:
            if layer_type != 'M': #卷积层
                layer = [nn.Conv3d(self.in_channels, out_channels = self.in_channels, kernel_size = (1, 3, 3), stride = 1, padding = 1), #卷积层
                                     #输入通道数                   输出特征数        卷积核大小，可以(x,y)    步长   ((上,下),(左,右)) 合并
                         nn.Conv3d(self.in_channels, out_channels = layer_type, kernel_size = 1, stride = 1, padding = 0), #深度可分离卷积 
                                    
                         nn.BatchNorm3d(num_features = layer_type), #BN层
                                         #特征数
                         nn.ReLU(inplace = True)]  #ReLU层 
                         #inplace=True改变输入的数据, 训练时要设立True，节省内存
                self.in_channels = layer_type #调整输入通道数
            else: #最大池化层
                layer = [nn.MaxPool3d(kernel_size = (1, self.pool_kernel_size, self.pool_kernel_size), stride = (1, self.pool_stride, self.pool_stride) )] #最大池化层
                                                                    #池化核大小                                       步长
                #输入通道数不变
            layers += layer

        #layers.append(nn.Conv2d(in_channels, out_channels = 512, kernel_size = 1, stride = 1)) #利用NiN控制通道数并增加线性，更加robust            
        layers.append(nn.AdaptiveAvgPool3d(output_size = (1, 1, 1))) #全局平均池化层，化简特征 -> 512 * 1 * 1, robust，且接受任意大小图片
                                               #输出大小      
        vgg_feature_net = nn.Sequential(*layers) #序列化,等价于列表
        
        return vgg_feature_net
        
    def _classifer_layers(self):
        """
        建立分类部分网络
        return vgg_classifer_net VGG分类部分
        """
        
        layers = []
        layers.append(nn.Dropout(p = 0.5)) #dropout层
                                   #概率值
        layers.append(nn.Linear(in_features = self.in_channels, out_features = self.class_num, bias = True)) #分类层(最后的全连接层)
                                   #输入特征数               输出特征数           偏移项 默认True
        #layers.append(nn.Softmax(dim = 0)) #softmax层，交叉熵里面包含，分类不需要softmax, 且对于非one-hot label多分类不适用
                             #dim = 0列求softmax        
        vgg_classifer_net = nn.Sequential(*layers) #序列化
        
        return vgg_classifer_net
           
    def forward(self, x):
        """
        前向传播
        :param: x 图片变量
        return out: 前向传播输出
        """
        feature = self.features(x) #获取x对应特征
        flat_feature = feature.view(feature.size(0), -1) #view类似resize，特征化为一维
        out = self.classifier(flat_feature) #获取x分类结果
        
        return out
  
    
  
#ResNet神经网络
        
#用于ResNet18和34的残差块，2个3x3的卷积优化为 2个(1*3 + 3*1)
class BasicBlock(nn.Module):
    """
    用于ResNet18和34的残差块, 2个3x3的卷积优化为 2个(1*3 + 3*1)
    """  
    expansion = 1 #输出扩大倍数

    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        """
        小残差块初始化
        :param: in_channels 卷积层输入通道数
        :param: out_channels 卷积层输出通道数
        :param: stride 卷积层步长
        :param: downsample 是否要调整, 为下采样网络结构
        """
        super(BasicBlock, self).__init__() # 等价与nn.Module.__init__()   运用nn.Module初始化
        #super为调用父类初始化
        
        self.downsample = downsample #是否要调整, 为下采样网络结构
        
        #中间输出部分
        self._make_midout = nn.Sequential( 
            #第一个卷积层堆，不一定same_padding, stride = 1是same
            #3*1卷积层
            nn.Conv3d(in_channels, out_channels, kernel_size = (1, 3, 1), stride = (1, stride, 1), padding = (0, 1, 0), bias = False),
                                                    #卷积核 1*3*1        深度1，上下stride,左右1      深度无，上下有，左右无
            nn.BatchNorm3d(out_channels), #对应BN层
            nn.ReLU(inplace = True), #对应ReLU层 
            
            #1*3卷积层                
            nn.Conv3d(out_channels, out_channels, kernel_size = (1, 1, 3), stride = (1, 1, stride), padding = (0, 0, 1), bias = False),
                                                    #卷积核 1*1*3      深度1，上下1, 左右stride   深度无，上下无，左右有
            nn.BatchNorm3d(out_channels), #对应BN层
            nn.ReLU(inplace = True), #对应ReLU层 
                                   
            #第二个卷积层堆, same_padding                                                                      
            nn.Conv3d(out_channels, self.expansion * out_channels, kernel_size = (1, 3, 1), stride = 1, padding = (0, 0, 1), bias = False),
                                                                        #卷积核 1*1*3                     深度无，上下有，左右无
            nn.BatchNorm3d(out_channels), #对应BN层
            nn.ReLU(inplace = True), #对应ReLU层 
            nn.Conv3d(self.expansion * out_channels, self.expansion * out_channels, kernel_size = (1, 1, 3), stride = 1, padding = (0, 1, 0), bias = False),
                                                                                        #卷积核 1*1*3                     深度无，上下无，左右有               
            nn.BatchNorm3d(self.expansion * out_channels) #对应BN层
            )               
            
    def forward(self, x):
        """
        前向传播
        :param: x 图片变量
        return out: 前向传播输出
        """
        residual = x
        midout = self._make_midout(x) #获得中间结果
        if self.downsample is not None: 
            residual = self.downsample(x)#修正残差
        inputs = residual + midout #获得ReLU层输入
        out = nn.ReLU(inplace=True)(inputs) #输出激活值

        return out


#用于ResNet50,101和152的残差块，1x1 + 3x3 + 1x1的卷积优化为1x1 + 3x1 + 1*3 + 1x1
class Bottleneck(nn.Module):
    """
    用于ResNet50,101和152的残差块，1x1 + 3x3 + 1x1的卷积优化为1x1 + 3x1 + 1*3 + 1x1
    """
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4 #输出扩大倍数

    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        """
        小残差块初始化
        :param: in_channels 卷积层输入通道数
        :param: out_channels 卷积层输出通道数
        :param: stride 卷积层步长
        :param: downsample 是否要调整, 为下采样网络结构
        """
        super(Bottleneck, self).__init__() # 等价与nn.Module.__init__()   运用nn.Module初始化
        #super为调用父类初始化
        
        self.downsample = downsample
        
        #残差
        self._make_midout = nn.Sequential( 
            #第一个卷积层, NiN, stride = 1
            nn.Conv3d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0,  bias = False),
            nn.BatchNorm3d(out_channels), #对应BN层
            nn.ReLU(inplace = True), #对应ReLU层
            
            #第二个卷积层堆, 不一定same_padding, stride = 1是same 
            #1*3*1卷积层
            nn.Conv3d(out_channels, out_channels, kernel_size = (1, 3, 1), stride = (1, stride, 1), padding = (0, 1, 0), bias = False),
                                                        #卷积核 1*3*1         深度1,上下stride,左右1      深度有,上下有，左右无
            nn.BatchNorm3d(out_channels), #对应BN层
            nn.ReLU(inplace = True), #对应ReLU层 

            #1*1*3卷积层                
            nn.Conv3d(out_channels, out_channels, kernel_size = (1, 1, 3), stride = (1, 1, stride), padding = (0, 0, 1), bias = False),
                                                    #卷积核 1*3           深度有, 上下1, 左右stride    深度无,上下无，左右有
            nn.BatchNorm3d(out_channels), #对应BN层
            nn.ReLU(inplace = True), #对应ReLU层 
 
            #第三个卷积层,  NiN, stride = 1                            
            nn.Conv3d(out_channels, out_channels * self.expansion, kernel_size = 1, stride = 1, padding = 0,  bias = False),
            nn.BatchNorm3d(out_channels * self.expansion), #对应BN层
            )
        
    def forward(self, x):
        """
        前向传播
        :param: x 图片变量
        return out: 前向传播输出
        """
        residual = x
        midout = self._make_midout(x) #获得中间结果
        if self.downsample is not None:
            residual = self.downsample(x)
        inputs = residual + midout #获得ReLU层输入
        out = nn.ReLU(inplace=True)(inputs) #输出激活值
        
        return out


class ResNets(nn.Module):
    """
    ResNets神经网络搭建，部分优化
    """
    
    cfg = {'ResNet_18':[BasicBlock, [2,2,2,2]], 'ResNet_34':[BasicBlock, [3,4,6,3]],
           'ResNet_50':[Bottleneck, [3,4,6,3]], 'ResNet_101':[Bottleneck, [3,4,23,3]], 'ResNet_152':[Bottleneck, [3,8,36,3]]}
                      #残差块调用类   对应个数
                      
    def __init__(self, ResNet_name, classes_num):
        """
        网络初始化
        :param: ResNet_name 选用的ResNet模型名字
        :param: classes_num 分类数
        """
        super(ResNets, self).__init__()# 等价与nn.Module.__init__()   运用nn.Module初始化
        #super为调用父类初始化
        
        self.ResNet_name = ResNet_name
        self.classes_num = classes_num
        self.block_class = self.cfg[ResNet_name][0] #对应残差块类型
        self.num_blocks = self.cfg[ResNet_name][1] #对应残差块个数
        self.expansion = self.block_class.expansion #对应膨胀系数
 
        self.in_channels = 64 #全局化输入通道数
        
        self.conv = nn.Sequential(
                   #初始卷积层集合(没加残差块) 调整为两个卷积组合 （用四个3*3 same 代替 7*7 same)
                   nn.Conv3d(in_channels = 3, out_channels = 64, kernel_size = (1,3,3), stride = 1, padding = (0,1,1), bias = False),
                   nn.BatchNorm3d(num_features = 64), #对应BN层
                   nn.ReLU(inplace = True), #对应ReLU层
                   nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size = (1,3,3), stride = 1, padding = (0,1,1), bias = False),
                   nn.BatchNorm3d(num_features = 64), #对应BN层
                   nn.ReLU(inplace = True), #对应ReLU层
                   nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size = (1,3,3), stride = 1, padding = (0,1,1), bias = False),
                   nn.BatchNorm3d(num_features = 64), #对应BN层
                   nn.ReLU(inplace = True), #对应ReLU层
                   nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size = (1,3,3), stride = 1, padding = (0,1,1), bias = False),
                   nn.BatchNorm3d(num_features = 64), #对应BN层
                   nn.ReLU(inplace = True), #对应ReLU层

                   )
        self.layer1 = self._make_layer(out_channels = 64, num_block = self.num_blocks[0], stride = 1) #特征大小不变
        self.layer2 = self._make_layer(128, self.num_blocks[1], 2) #特征大小/2
        self.layer3 = self._make_layer(256, self.num_blocks[2], 2) #特征大小/2
        self.layer4 = self._make_layer(512, self.num_blocks[3], 2) ##特征大小/2，残差块堆网络
        
        self.GAP = nn.AdaptiveAvgPool3d((1, 1, 1)) #全局池化，将特征化为（512*expansion）*1*1
        self.dropout = nn.Dropout(p = 0.2) #dropout层
        self.linear = nn.Linear(512 * self.expansion, self.classes_num) #全连接层分类网络
                             #输入为最后输出 * expansion   分类数
        
    def _make_layer(self, out_channels, num_block, stride):
        """
        搭建对应的残差块集合网络
        输入通道数类内全局化, 因为in_channels到第二块才改为out_channels * expansion, 为方便定义将其全局化
        :param: out_channels 输出通道数
        :param: num_block 对应残差块个数
        :param: stride 步长
        return: residuals_net 残差块集合网络
        """
        downsample = None
        # x要与残差的维度相同(大小和通道数), 如果不相同，需要添加卷积 + BN来变换为同一维度   
        if stride != 1 or self.in_channels != self.expansion * out_channels: #只有第一块需要downsample
        #没有same pad 大小不同           x维度与残差通道数不一致
            downsample = nn.Sequential(
                #修正用NiN卷积层
                nn.Conv3d(self.in_channels, self.expansion * out_channels, kernel_size = 1, stride = (1,stride,stride), bias = False),
                                                    #修正通道数                                修正大小              
                nn.BatchNorm3d(self.expansion * out_channels) #对应BN层
            )  #x通过卷积 + BN修正来转换后加入
            
        layers = []
        layers.append(self.block_class(self.in_channels, out_channels, stride, downsample)) #一开始步长不等于1且可能 downsample
        self.in_channels = out_channels * self.expansion #第二块后输入为 out_channels * expansion，因为之后为same padding所以不变
        for i in range(1, num_block): #遍历
            layers.append(self.block_class(self.in_channels, out_channels)) #加一个残差块,    之后为same padding, 且不会downsample
        
        residuals_net = nn.Sequential(*layers) #序列化
        
        return residuals_net  

    def forward(self, x):
        """
        前向传播
        :param: x 图片变量
        return out: 前向传播输出
        """

        out = self.conv(x) #残差前卷积层 
        out = self.layer1(out) #第一个残差块堆
        out = self.layer2(out) #第二个残差块堆
        out = self.layer3(out) #第三个残差块堆
        out = self.layer4(out) #第四个残差块堆
        out = self.GAP(out) #卷积池化层
        out = self.dropout(out) #dropout层
        out = out.view(out.size(0), -1) #view类似resize，特征化为一维
        out = self.linear(out) #fc分类层
      
        return out
 
        
"""
通用模型初始化
"""
def initialize_weights(model):
    """
    模型初始化
    :param:model 输入模型 可以用model.apply(initialize_weights)调用
    """
    for module in model.modules(): #模型中的所有模式，包含总，序列，层
        
        if isinstance(module, nn.Conv3d): #卷积层
            n_conv = module.kernel_size[0] * module.kernel_size[1] * module.out_channels #卷积权重元素个数
            module.weight.data.normal_(0, math.sqrt(2. / n_conv)) #卷积核初始化 正态随机数, 限制标准差
            if module.bias is not None: #有偏移项
                module.bias.data.zero_() #偏移项初始化 = 0
                
        elif isinstance(module, nn.BatchNorm3d): #BN层
            module.weight.data.fill_(1) #归一化权重 == 标准差, 初始化 = 1
            module.bias.data.zero_() #归一化偏移项 == 均值, 初始化 = 0
            
        elif isinstance(module, nn.Linear): #全连接层
            n_fc = module.in_features * module.out_features #全连接层权重个数
            module.weight.data.normal_(0, math.sqrt(2. / n_fc)) #全连接权重正态初始化
            module.bias.data.zero_()#偏移项初始化 = 0
        

"""
获取变量名对应模型
"""
def get_model(model_name, classes):
    """
    :param: classes 分类数
    :param: model_name 模型名字
    return model 模型类  
    """
    model_dict = {'VGG': VGGs, 'ResNet': ResNets, 'AlexNet': AlexNet, 'LeNet': LeNet}
    if '_' in model_name: #有不同版本的模型
        model_type = model_name.split('_')[0] #获取模型类型
        return model_dict[model_type](model_name, len(classes))
                #对应模型类             模型名       分类数
    else:#没有不同版本的模型
        model_type = model_name#获取模型类型
        return model_dict[model_type](len(classes))
                #对应模型类             分类数
                
                