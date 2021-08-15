import torch 
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt 
from PIL import Image

TRAIN_DIR = "dataset/renamed/"
IMAGE_SIZE = 160

train_data = torchvision.datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        torchvision.transforms.ToTensor(),
    ])
)

train_data = torch.utils.data.DataLoader(
    train_data,
    batch_size=1,
    shuffle=True
)

def train():
    # for data in tqdm(train_data):
    #     x,target = data 
    #     print(target)
    for data in train_data:
        x,target = data
        print(target)
    pass 

train()