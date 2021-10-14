import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
import torchvision
import torchvision.utils as vutils

def init_data(batch_size=128, data_dir='./dataset'):
    # transformation given
    transform = transforms.Compose([transforms.Resize(64), 
                                    transforms.CenterCrop(64), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataset = torchvision.datasets.CelebA(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader
