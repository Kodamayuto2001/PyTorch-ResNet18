import torch 
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self,input_channel):
        self.inplanes = 64
        super(ResNet,self).__init__()

        #   Layer Name : conv1
        self.conv1 = nn.Conv2d(
            input_channel,
            self.inplanes,
            kernel_size=(7,7),
            stride=(2,2),
            padding=(3,3)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)   #   C言語のポインタみたいな処理
        self.maxpool = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=(2,2),
            padding=(1,1),
        )

        #   Layer Name : conv2_x

    def _make_layer(self):
        pass 

    