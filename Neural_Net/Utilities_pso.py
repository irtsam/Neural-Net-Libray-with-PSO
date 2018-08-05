#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:48:14 2018

@author: iot
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1= nn.Linear(32*32, 120)
        self.fc2= nn.Linear(120, 84)
        self.fc3= nn.Linear(84, 10)
       
        
        
        """Thus our neural network parameters have been defined"""
    def forward(self,x):
        
        
        x= f.sigmoid(self.fc1(x))
        #print(x.size())
        
        x= f.sigmoid(self.fc2(x))
        #x = f.dropout(x, p=0.1,inplace= True)
        #print(x.size())
        x= f.sigmoid(self.fc3(x))
     
        
        return x
        
