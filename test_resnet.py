import os 
import cv2
import sys 
import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
import torchvision.transforms as transforms 

#  kernel_sizeが3x3，padding=stride=1のconvは非常によく使用するので、関数で簡単に呼べるようにする
def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                    padding=dilation, groups=groups, bias=True,
                    dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)


class BasicBlock(nn.Module):
    #  Implementation of Basic Building Block

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity_x = x  # hold input for shortcut connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity_x = self.downsample(x)

        out += identity_x  # shortcut connection
        return self.relu(out)


class ResidualLayer(nn.Module):
    
     def __init__(self, num_blocks, in_channels, out_channels, block=BasicBlock):
         super(ResidualLayer, self).__init__()
         downsample = None
         if in_channels != out_channels:
             downsample = nn.Sequential(
                 conv1x1(in_channels, out_channels),
                 nn.BatchNorm2d(out_channels)
         )
         self.first_block = block(in_channels, out_channels, downsample=downsample)
         self.blocks = nn.ModuleList(block(out_channels, out_channels) for _ in range(num_blocks))

     def forward(self, x):
         out = self.first_block(x)
         for block in self.blocks:
             out = block(out)
         return out

class ResNet18(nn.Module):
    
   def __init__(self, num_classes):
       super(ResNet18, self).__init__()
       self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
       self.bn1 = nn.BatchNorm2d(64)
       self.relu = nn.ReLU(inplace=True)
       self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
       self.layer1 = ResidualLayer(2, in_channels=64, out_channels=64)
       self.layer2 = ResidualLayer(2, in_channels=64, out_channels=128)
       self.layer3 = ResidualLayer(
           2, in_channels=128, out_channels=256)
       self.layer4 = ResidualLayer(
           2, in_channels=256, out_channels=512)
       self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
       self.fc = nn.Linear(512, num_classes)

   def forward(self, x):
       out = self.conv1(x)
       out = self.bn1(out)
       out = self.relu(out)
       out = self.maxpool(out)

       out = self.layer1(out)
       out = self.layer2(out)
       out = self.layer3(out)
       out = self.layer4(out)

       out = self.avg_pool(out)
       out = out.view(out.size(0), -1)
       out = self.fc(out)

       return out