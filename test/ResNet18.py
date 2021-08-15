import torch 
import cv2

class Net(torch.nn.Module):
    def __init__(self,channel_in,channel_out):
        super(Net,self).__init__()
        channel = channel_out
        self.conv1 = torch.nn.Conv2d(channel_in,channel,kernel_size=(1,1))

    def forward(self,x):
        print(x.size())
        return x


