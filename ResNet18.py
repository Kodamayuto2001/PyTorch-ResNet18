import enum
import os 
import cv2
import torch 
import numpy as np
import torchvision
import torch.nn as nn
from torchvision.transforms.transforms import Resize
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from torchvision import transforms

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.num_classes = num_classes
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(128*512*2*2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        x = self.avgpool(x)
        # print(x.shape)
        
        x = x.view(x.size(0), -1)
        # print(x.shape)
        logits = nn.Linear(x.shape[1],self.num_classes)(x)
        # logits = self.fc(x)
        # logits = nn.Linear(x.shape[0] * x.shape[1],self.num_classes)(x)
        probas = F.log_softmax(logits, dim=1)
        return probas

    
class Training:
    def __init__(
        self,
        train_root_dir = "dataset/train/",
        test_root_dir = "dataset/test/",
        batch_size=128,
        num_classes=3,
        grayscale=True,
        lr = 0.001,
        image_size = 256,
        num_epoch = 10,
        pt_name = "nn.pt",
        loss_png = "loss.png",
        acc_png = "acc.png"
    ):
        self.model = ResNet(block=BasicBlock, 
                   layers=[2, 2, 2, 2],
                   num_classes=num_classes,
                   grayscale=grayscale)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=lr)
        self.image_size = image_size
        self.epoch = epoch 
        self.pt_name = pt_name
        self.loss_png = loss_png
        self.acc_png = acc_png
        self.num_epoch = num_epoch
        
        if grayscale == True:
            train_data = torchvision.datasets.ImageFolder(
                root=train_root_dir,
                transform=transforms.Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.ToTensor(),
                    transforms.Grayscale()
                ])
            )

            self.train_data = torch.utils.data.DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True
            )

            test_data = torchvision.datasets.ImageFolder(
                root=test_root_dir,
                transform=transforms.Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.ToTensor(),
                    transforms.Grayscale()
                ])
            )

            self.test_data = torch.utils.data.DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False
            )
        else:
            train_data = torchvision.datasets.ImageFolder(
                root=train_root_dir,
                transform=transforms.Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.ToTensor()
                ])
            )

            self.train_data = torch.utils.data.DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True
            )

            test_data = torchvision.datasets.ImageFolder(
                root=test_root_dir,
                transform=transforms.Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.ToTensor()
                ])
            )

            self.test_data = torch.utils.data.DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False
            )

    # def train(self):
    #     for data in tqdm(self.train_data):
    #         x,target = data 
    #         output = self.model(x)
    #         loss = F.cross_entropy(output,target)
    #         loss.backward()
    #         self.optimizer.step()
    #     return loss.detach().numpy() 

    # def test(self):
    #     self.model.eval()
    #     total = 0
    #     correct = 0
    #     with torch.no_grad():
    #         for data in tqdm(self.test_data):
    #             x,target = data 
    #             output = self.model(x)
    #             _,p = torch.max(output.data,1)
    #             total += target.size(0)
    #             correct += (p == target).sum().item()
    #     percent = 100*correct/total 
    #     return percent
                
    # def maesyori(self,path,label):
    #     img     =   cv2.imread(path)
    #     img     =   cv2.resize(img,(self.image_size,self.image_size))
    #     img     =   np.reshape(img,(1,self.image_size,self.image_size))
    #     img     =   np.transpose(img,(1,2,0))
    #     img     =   torchvision.transforms.ToTensor()(img)
    #     img     =   img.unsqueeze(0)
    #     label   =   torch.Tensor([label]).long()
    #     return img,label

    def train(self,epoch):
        for batch_idx,(x,targets) in enumerate(self.train_data):
            output = self.model(x)
            # print(logits)
            # print(targets)
            cost = F.nll_loss(output,targets)
            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

            print("Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f"%(epoch+1,self.num_epoch,batch_idx,len(self.train_data),cost))

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

    def showData(self):
        for x,y in self.train_data:
            # print(x.shape)
            print(y)

if __name__ == "__main__":
    # epoch = 10
    # ai = Training(epoch=epoch)
    # loss = []
    # acc = []
    # for e in range(epoch):
    #     loss.append(ai.train())
    #     acc.append(ai.test())

    # ai.save_loss_png(loss)
    # ai.save_acc_png(acc)
    # ai.save_model()
    
    epoch = 10
    ai = Training(num_epoch=epoch,lr=0.001,grayscale=False)
    # ai.showData()
    for e in range(epoch):
        ai.train(e)
