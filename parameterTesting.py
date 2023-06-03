from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
from typing import Tuple
from torch.utils.data import Dataset,DataLoader
# from sklearn.metrics import confusion_matrix, top_k_accuracy_score
import torchvision                                                       
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.transforms import ToTensor
from numpy import random as rd
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from torch.utils.data.sampler import SubsetRandomSampler
import time

print(torch.cuda.get_device_name(0))
# transform method
transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5), (0.5)),
    ])

# train data
trainData = datasets.FashionMNIST(root="./",
                                  train=True,
                                  transform=transform,
                                  download=True
                                  )
trainLoad = DataLoader(trainData, 
                       batch_size=30, 
                       shuffle=True, 
                       drop_last=False
                       )
# test data
testData = datasets.FashionMNIST(root="./",
                                  train=False,
                                  transform=transform,
                                  download=True
                                  )
testLoad = DataLoader(testData, 
                     batch_size=30, 
                     shuffle=True,
                     drop_last=False
                     )




class ResidualBlock(nn.Module):
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        
        # print("-new block-")
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
        """ The channel mismatch happens here, with this con2d layer
            This is the block made when in_channels=64 and out_channels=128"""
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=stride, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels*self.expansion)
        )

        # downsample for forwarding
        # short = []
        # if stride != 1 or in_channels != out_channels * self.expansion:
        #     short.append(nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, padding=0))
        #     short.append(nn.BatchNorm2d(out_channels * self.expansion))
        # self.short = nn.Sequential(*short)
        # self.relu = nn.ReLU()

        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels=out_channels
        
        # print("resblock in: " + str(in_channels))
        # print("resblock out: " + str(out_channels))

    def forward(self, x):
        # out = torch.cat([self.conv1(residual), self.conv2(residual)], 1)
        # residual = self.short(residual)
        # out = self.relu(out + residual)
        
        residual = x
        
        x = self.conv1(x)
        # print(x.size())
        
        """ It breaks in making this layer, I think
            At this point the tenor going into this is torch.Size([30, 128, 7, 7])"""
        x = self.conv2(x)

        x = self.conv3(x)

        if self.downsample != None:
            residual = self.downsample(residual)
            
        x += residual
        x = self.relu(x)
        return x


# class for ResNet model that extend from nn.Module
class Resnet(nn.Module):
    
    # initialize the resnet model with inputted block type, list of blockNum 
    def __init__(self, block, blockList, input_num, output_num):
        super(Resnet, self).__init__()
        
        self.in_channels = 16
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_num, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 3, 4, 6, 3 blocks
        # self.block0 = self._make_layer(block,   inChannels=64,  outChannels=64, blocksNum=blockList[0], stride=1)
        # self.block1 = self._make_layer(block,  inChannels=256, outChannels=128, blocksNum=blockList[1], stride=1)
        # self.block2 = self._make_layer(block,  inChannels=512, outChannels=256, blocksNum=blockList[2], stride=1)
        # self.block3 = self._make_layer(block, inChannels=1024, outChannels=512, blocksNum=blockList[3], stride=2)
        
        # print("---block0layer---")
        self.block0 = self._make_layer(block,  out_channels=16, blocksNum=blockList[0], stride=1)
        # print("---block1layer---")
        self.block1 = self._make_layer(block,  out_channels=32, blocksNum=blockList[1], stride=2)
        # print("---block2layer---")
        self.block2 = self._make_layer(block,  out_channels=64, blocksNum=blockList[2], stride=2)
        # print("---block3layer---")
        self.block3 = self._make_layer(block,  out_channels=128, blocksNum=blockList[3], stride=2)
        
        # apply 2D adaptive average pooling from 1 input to 1 plane
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpool = nn.AvgPool2d(1, 1)
        
        # flatten the data into 1 dimension ( apparently not useful)
        self.flatten = nn.Flatten()
        
        # apply dropout to output with 60% percent chance
        self.drop = nn.Dropout(0.6)
        
        # connect 512 input nodes into 10 output nodes since the last layer ended with the expanded out channel
        self.fc = nn.Linear(128*4, output_num)

    # helper function that adds layer by layer along with the res block
    def _make_layer(self, block: ResidualBlock, out_channels, blocksNum, stride):
        downn_sample = None
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downn_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        # layer that change the number of out channel, in will be inputed out
        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downn_sample))
        self.in_channels = out_channels * block.expansion
        # print("current inplanes: " + str(self.in_channels))
        
        # connected large output to smaller out 
        for _ in range(1, blocksNum):
            layers.append(block(self.in_channels, out_channels))
            
        # print("layers made")

        return nn.Sequential(*layers)
    
    # forward function 
    def forward(self, x: ToTensor):
        # initial the first convelution
        x = self.conv1(x)
        
        # making block
        x = self.block0(x)
        # print("done block0")
        x = self.block1(x)
        # print("done block1")
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.avgpool(x)
        # x = self.drop(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def boilerplate(ep, lr, wdr, mom):    
    # number of epoch
    epochNum=ep
    # learning rate
    learningRate = lr
    # weight decay
    weightDecayRate = wdr
    # momentum
    momentumAmount = mom
    # setting up the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # setting up the model
    # block using the ResidualBlock
    # blockNums using the inputted list
    # input_num of 1 for gray scaled, 3 for color
    # output_num of 10 for 10 classes
    model = Resnet(ResidualBlock, [3, 4, 6, 3], 1, 10).to(device)
    # loss
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecayRate)
    total_step = len(trainLoad)
    # print(model)
    for epoch in range(epochNum):
        for i, (images, labels) in enumerate(trainLoad):
            # move tensor to device
            images = images.to(device)
            labels = labels.to(device)
            
            # forward the output and calculate loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # backward the output and perform optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # deallocation
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()
            
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochNum, loss.item()))
        
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testLoad:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
        
       # print('Accuracy of the network on the {} validation images: {} %'.format(10000, 100 * correct / total))
    return correct, total



for epoch in range(1, 41):
         trainData = datasets.FashionMNIST(root="./",
                                  train=True,
                                  transform=transform,
                                  download=False
                                  )
         trainLoad = DataLoader(trainData, 
                                batch_size=512, 
                            shuffle=True, 
                               drop_last=False
                            )
         testData = datasets.FashionMNIST(root="./",
                                  train=False,
                                  transform=transform,
                                  download=False
                                  )
         testLoad = DataLoader(testData, 
                     batch_size=512, 
                     shuffle=True,
                     drop_last=False
                     )
         rnResults = open('results.txt', 'a')
         start_time = time.time()
         correct, total = boilerplate(epoch, .01, .001, .9)
         runTime = time.time() - start_time
         hours = int(np.floor(runTime / 3600))
         mins = int(np.floor((runTime - (hours * 3600)) /60))
         secs = ((runTime - (hours * 3600)) - (mins * 60))
         rnResults.write("Epoch: " + str(epoch) +"\nBatch: " + str(512) + "\n")
         rnResults.write('Accuracy of the network on the {} validation images: {} %'.format(10000, 100 * correct / total))
         rnResults.write("\nRuntime: " + str(hours) + ":" + str(mins) + ":" + str(secs) +"\n")
         rnResults.write("\n")
         rnResults.write("------------------END EPOCH " + str(epoch) + "------------------\n\n------------------BEGIN EPOCH " + str(epoch + 1) +"------------------\n")
         rnResults.close()
exit(1)