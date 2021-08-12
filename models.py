# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 17:13:56 2021

@author: black
"""
import torch
import torch.nn.functional as F
import torchvision as tv
from torchvision import models
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import cv2
import random

class DenseNet(nn.Module):
    def __init__(self,num_classes = 2):
        super(DenseNet,self).__init__()
        net = models.densenet201(pretrained = True)
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(
                nn.Linear(1920, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_classes),
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
    
class AlexNet(nn.Module):
    def __init__(self,num_classes = 2):
        super(AlexNet,self).__init__()
        net = models.alexnet(pretrained = True)
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(
                nn.Linear(9216, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_classes),
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
    
class googleNet(nn.Module):
    def __init__(self,num_classes = 2):
        super(googleNet,self).__init__()
        net = models.googlenet(pretrained = True)
        net.fc = nn.Sequential()
        self.features = net
        self.fc = nn.Sequential(
                nn.Linear(1024,2)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    
class VGGNet(nn.Module):
    def __init__(self,num_classes = 2):
        super(VGGNet,self).__init__()
        net = models.vgg16(pretrained = True)
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_classes),
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x