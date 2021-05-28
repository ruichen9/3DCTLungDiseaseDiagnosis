import torch
import torch.nn as nn
from torch.utils import data
from torch import optim
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as ptl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import *
from models.resnet18 import *
from pytorch_metric_learning import losses
import warnings
warnings.filterwarnings("ignore")



class my_ptlframe(ptl.LightningModule):
    def __init__(self):
        super().__init__()
        self.mymodel = generate_model()
        self.learning_rate = 1e-4
        self._automatic_optimization: bool = True
        print(self.parameters())
    def forward(self,x):
        return(self.mymodel(x))
    def training_step(self, batch, batch_idx):
        x,y = batch
        x=torch.tensor(x,dtype=torch.float32)
        embedding=self.forward(x)
        loss_fn = losses.ContrastiveLoss()
        loss = loss_fn(embedding, y)
        self.log('train_loss',
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True)
        return {'loss':loss}
    def validation_step(self, batch, batch_idx):
        image, label = batch
        image = torch.tensor(image, dtype=torch.float32)
        embedding = self.forward(image)
        #print(pred)
        loss_fn = losses.ContrastiveLoss()
        loss = loss_fn(embedding, label)
        self.log('val_loss', 
                loss, 
                on_step=True, 
                on_epoch=True, 
                prog_bar=True, 
                logger=True)
    def train_dataloader(self):
        my_traindataloader = train_dataloader
        return my_traindataloader
    def val_dataloader(self):
        my_validationdataloader = val_dataloader
        return my_validationdataloader
    def configure_optimizers(self):
        weight_decay = 1e-3  # l2正则化系数
        # 假如有两个网络，一个encoder一个decoder
        optimizer = optim.Adam(
                        self.parameters(),
                        lr=self.learning_rate,
                        weight_decay=weight_decay)
        # 同样，如果只有一个网络结构，就可以更直接了
        # optimizer = optim.Adam(
        #             self.parameters(), 
        #             lr=self.learning_rate, 
        #             weight_decay=weight_decay)
        # 我这里设置2000个epoch后学习率变为原来的0.5，之后不再改变
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.3)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict
    @property
    def automatic_optimization(self) -> bool:
        return self._automatic_optimization
    @automatic_optimization.setter
    def automatic_optimization(self, automatic_optimization: bool) -> None:
        self._automatic_optimization = automatic_optimization


def generate_model():
    model = ResNet(block=BasicBlock,
            layers=[2,2,2,2],
            block_inplanes=get_inplanes(),
            n_input_channels=1,
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            shortcut_type='B',
            widen_factor=1.0,
            embedding_size=128)
    return model


if __name__ == '__main__':

    train_dataloader, val_dataloader, test_dataloader = data_get('../Data', batch_size=4)
    
    tb_logger = pl_loggers.TensorBoardLogger('./logs/')

    checkpoint_callback=ModelCheckpoint(monitor='val_loss',
                                        save_last=True,
                                        save_top_k=-1,
                                        dirpath='./checkpoints',
                                        filename='resnet18metric-{epoch:02d}-{val_loss:.2f}')

    trainer = Trainer(gpus=[0],
                    precision=32,
                    check_val_every_n_epoch=1,
                    callbacks=[checkpoint_callback],
                    accumulate_grad_batches=8,
                    min_epochs=1,
                    max_epochs=10,
                    logger=tb_logger)


    my_ptl= my_ptlframe()

    a = trainer.fit(my_ptl)