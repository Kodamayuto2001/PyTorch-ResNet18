import torch 
import torchvision

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
