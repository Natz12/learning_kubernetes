# event vs no event classification for GOES 2hrs data with cnn 15*15 km^2 array size: 32*32
import glob
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from torch.nn import (AvgPool2d, BatchNorm2d, Conv2d, CrossEntropyLoss,
                      Dropout, Linear, MaxPool2d, Module, ReLU, Sequential,
                      Softmax)
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.loss import BCELoss
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")

def double_conv(in_channels, out_channels, kernel_size=3, padding=1):
    return Sequential(
        Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        BatchNorm2d(out_channels),
        ReLU(inplace=True),
        Conv2d(out_channels, out_channels, kernel_size, padding=padding),
        BatchNorm2d(out_channels),
        ReLU(inplace=True)
    )  
     
def last_conv(in_channels, out_channels, kernel_size=1, padding=0):
    return Sequential(
        Conv2d(in_channels, out_channels, kernel_size, padding=padding),
    )   
    
class UnetSegmentation(Module):
    def __init__(self):
        """
        """
        super().__init__()
        
        # self.dconv_down1 = double_conv(66, 128)
        self.dconv_down1 = double_conv(126, 128)
        self.dconv_down2 = double_conv(128, 256)
        self.dconv_down3 = double_conv(256, 512)
        # self.dconv_down4 = double_conv(512, 1024)        
        self.last_conv = last_conv(512, 128)
        self.maxpool = MaxPool2d(2)
        self.linear_layers = Sequential(
            Linear(in_features=2048,
                out_features=512,
                bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Linear(in_features=10000,
            #     out_features=1000,
            #     bias=True),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            Linear(in_features=512,
                out_features=2,
                bias=True),
            Sigmoid()
        )
          
    def forward(self, x):
        x = x.float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) 
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        # conv4 = self.dconv_down4(x) # 1024
        # x = self.maxpool(conv4) 
        x = self.last_conv(x)
        x = x.view(x.shape[0],1, -1) 
        x = self.linear_layers(x) 
  
        return x.squeeze(1)
