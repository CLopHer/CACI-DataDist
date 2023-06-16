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
import sys
f = open("testout.txt", 'w')
sys.stdout = f
# transform method
transform = transforms.Compose([
            transforms.ToTensor(),
    ])

# train data
trainData = datasets.FashionMNIST(root="./",
                                  train=True,
                                  transform=transform,
                                  download=True
                                  )
trainLoad = DataLoader(trainData, 
                       batch_size=512, 
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
                     batch_size=512, 
                     shuffle=True, 
                     drop_last=False
                     )

class ResidualBlock(nn.Module):
    
    expansion = 4   # factor by which to expand the number of features per block
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        
        # First block's convolutional layer with a batch normalization and RELU activation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # Second block's convolutional layer with batch normalization and RELU activation
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=stride, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
        # Third block's convolutional layer with batch normalization
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels*self.expansion)
        )

        #Finishing layers, with a downsample and activation
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels=out_channels

    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
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
        
        self.in_channels = 16  # Standard factor of feature channels to expand each block from
        
        # First convulotion layer with batch normalization, ReLU activation, and max pooling as this is the first layer from input
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_num, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Calling of the make layer functions to build middle of the model
        self.block0 = self._make_layer(block,  out_channels=16, blocksNum=blockList[0], stride=1)
        self.block1 = self._make_layer(block,  out_channels=32, blocksNum=blockList[1], stride=2)
        self.block2 = self._make_layer(block,  out_channels=64, blocksNum=blockList[2], stride=2)
        self.block3 = self._make_layer(block,  out_channels=128, blocksNum=blockList[3], stride=2)
        
        # apply 2D adaptive average pooling from 1 input to 1 plane
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpool = nn.AvgPool2d(1, 1)
        
        # flatten the data into 1 dimension
        self.flatten = nn.Flatten()
        
        # apply dropout to output with 60% percent chance
        self.drop = nn.Dropout(0.6)
        
        # connect 2048 input nodes into 10 output nodes
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
        
        # connected large output to smaller out 
        for _ in range(1, blocksNum):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    # forward function 
    def forward(self, x: ToTensor):
        x = self.conv1(x)
        
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.avgpool(x)
        x = self.drop(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
# number of epoch
epochNum=30
# learning rate
learningRate = .005
# weight decay
weightDecayRate = .001
# momentum
momentumAmount = .5
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

#count
count = 0

epochs = [e for e in range(epochNum)]
losses = []
accuracies = []
lossGraph =[]
predictions = []
label_list = []
start_time = time.time()
for epoch in range(epochNum):
    for i, (images, labels) in enumerate(trainLoad):
        # move tensor to device
        images = images.to(device)
        labels = labels.to(device)
        
        # forward the output and calculate loss
        outputs = model(images)
        loss = criterion(outputs, labels)
        losses.append(loss.data)
        
        # backward the output and perform optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # deallocation
        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()
    lossGraph.append(loss.item())
    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochNum, loss.item()))
    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testLoad:
            images = images.to(device)
            labels = labels.to(device)
            label_list.append(labels)
            outputs = model(images)
            predicted = torch.max(outputs, 1)[1].to(device)
            predictions.append(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            del images, labels, outputs
    accuracy = 100 * correct / total
    accuracies.append(accuracy)
    print('Accuracy of the network on the {} validation images: {} %'.format(10000, accuracy))

runTime = time.time() - start_time
hours = int(np.floor(runTime / 3600))
mins = int(np.floor((runTime - (hours * 3600)) /60))
secs = ((runTime - (hours * 3600)) - (mins * 60))
print("\nRuntime: " + str(hours) + ":" + str(mins) + ":" + str(secs) +"\n")
i = 0
plt.plot(epochs, lossGraph)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Epochs vs Loss")
plt.show()
plt.savefig("EpochsvLoss.png")
i = 0
for x in accuracies:
    accuracies[i] = x.cpu()
    i+=1


plt.plot(epochs, accuracies)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Epochs vs Accuracy")
plt.show()
plt.savefig("EpochsvAccuracy.png")
f.close()
from itertools import chain 

predictions_list = [predictions[i].tolist() for i in range(len(predictions))]
labels_list = [label_list[i].tolist() for i in range(len(label_list))]
predictions_list = list(chain.from_iterable(predictions))
labels_list = list(chain.from_iterable(labels_list))