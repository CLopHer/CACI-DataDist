import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

''' ResNet '''

class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
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

class ResNet(nn.Module):
     # initialize the resnet model with inputted block type, list of blockNum 
    def __init__(self, block, blockList, input_num=1, output_num=10):
        super(ResNet, self).__init__()
        
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
    def _make_layer(self, block: BasicBlock, out_channels, blocksNum, stride):
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


def ResNet34(channel, num_classes):
    return ResNet(BasicBlock, [3,4,6,3], input_num=channel, output_num=num_classes)