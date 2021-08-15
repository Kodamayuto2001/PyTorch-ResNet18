import torch 
import torch.nn as nn

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
        out = self.fc(out)

        return out 

if __name__ == "__main__":
    inputs = torch.zeros((5,3,128,128))
    model = ResNet18(num_classes=10)
    outputs = model(inputs)
    print(outputs.size())
    print(outputs.data.size())

    