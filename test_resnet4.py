import os 
import cv2
import torch 
import numpy as np
import torchvision
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from torchvision import transforms

def conv3x3(input_channels,output_channels,stride=(1,1),groups=1,dilation=(1,1)):
    return nn.Conv2d(
        input_channels,
        output_channels,
        kernel_size=(3,3),
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )

def conv1x1(input_channels,output_channels,stride=(1,1)):
    return nn.Conv2d(input_channels,output_channels,kernel_size=(1,1),bias=False)

class BasicBlock(nn.Module):
    def __init__(self,input_channels,output_channels,stride=(1,1),downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(input_channels,output_channels,stride)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(output_channels,output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.downsample = downsample

    def forward(self,x):
        identity = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
    
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResidualLayer(nn.Module):
    def __init__(self,num_blocks,input_channels,output_channels,block=BasicBlock):
        super(ResidualLayer,self).__init__()
        downsample = None 
        if input_channels != output_channels:
            downsample = nn.Sequential(
                conv1x1(input_channels,output_channels),
                nn.BatchNorm2d(output_channels)
            )
        self.first_block = block(input_channels,output_channels,downsample=downsample)
        self.blocks = nn.ModuleList(block(output_channels,output_channels) for _ in range(num_blocks))

    def forward(self,x):
        out = self.first_block(x)
        for block in self.blocks:
            out = block(out)
        return out

class ResNet18(nn.Module):
    def __init__(self,num_classes):
        self.inplanes = 64
        super(ResNet18,self).__init__()

        #   Layer Name : conv1
        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=(7,7),
            stride=(2,2),
            padding=(3,3)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=(2,2),
            padding=(1,1),
        )

        
        self.layer1 = ResidualLayer(2,input_channels=64,output_channels=64)
        self.layer2 = ResidualLayer(2,input_channels=64,output_channels=128)
        self.layer3 = ResidualLayer(2,input_channels=128,output_channels=256)
        self.layer4 = ResidualLayer(2,input_channels=256,output_channels=512)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_classes)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0),-1)
        out = F.log_softmax(out,dim=1)

        return out 
    
class Training:
    def __init__(
        self,
        train_dir = "dataset/train/",
        test_dir = "dataset/test/",
        num_classes=3,
        lr = 0.001,
        image_size = 150,
        epoch = 10,
        pt_name = "nn.pt",
        loss_png = "loss.png",
        acc_png = "acc.png"
    ):
        self.model = ResNet18(num_classes=num_classes)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=lr)
        self.image_size = image_size
        self.epoch = epoch 
        self.pt_name = pt_name
        self.loss_png = loss_png
        self.acc_png = acc_png

        train_data = torchvision.datasets.ImageFolder(
            root=train_dir,
            transform=transforms.Compose([
                transforms.Resize((image_size,image_size)),
                transforms.ToTensor()
            ])
        )
        self.train_data = torch.utils.data.DataLoader(
            train_data,
            batch_size=1,
            shuffle=True
        )

        test_data = torchvision.datasets.ImageFolder(
            root=test_dir,
            transform=transforms.Compose([
                transforms.Resize((image_size,image_size)),
                transforms.ToTensor()
            ])
        )
        self.test_data = torch.utils.data.DataLoader(
            test_data,
            batch_size=1,
            shuffle=False
        )

    def train(self):
        for data in tqdm(self.train_data):
            x,target = data 
            output = self.model(x)
            loss = F.nll_loss(output,target)
            loss.backward()
            self.optimizer.step()
        return loss 

    def test(self):
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data in tqdm(self.test_data):
                x,target = data 
                output = self.model(x)
                _,p = torch.max(output.data,1)
                total += target.size(0)
                correct += (p == target).sum().item()
        percent = 100*correct/total 
        return percent
                
    def maesyori(self,path,label):
        img     =   cv2.imread(path)
        img     =   cv2.resize(img,(self.image_size,self.image_size))
        img     =   np.reshape(img,(1,self.image_size,self.image_size))
        img     =   np.transpose(img,(1,2,0))
        img     =   torchvision.transforms.ToTensor()(img)
        img     =   img.unsqueeze(0)
        label   =   torch.Tensor([label]).long()
        return img,label

    def save_loss_png(self,loss):
        plt.figure()
        plt.plot(range(1,self.epoch+1),loss,label="trainLoss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(self.loss_png)

    def save_acc_png(self,acc):
        plt.figure()
        plt.plot(range(1,self.epoch+1),acc,label="all-acc")
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(self.acc_png)

    def save_model(self):
        torch.save(self.model.state_dict(),self.pt_name)

if __name__ == "__main__":
    epoch = 10
    ai = Training(epoch=epoch)
    loss = []
    acc = []
    for e in range(epoch):
        loss.append(ai.train())
        acc.append(ai.test())

    ai.save_loss_png(loss)
    ai.save_acc_png(acc)
    ai.save_model()