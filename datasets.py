import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
import torchvision
import torchvision.utils as vutils

def init_data(data_dir='./dataset'):
    # transformation given
    transform = transforms.Compose([transforms.Resize(64), 
                                    transforms.CenterCrop(64), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataset = torchvision.datasets.CelebA(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    return dataloader


def save_image_list(dataset, real):
    if real:
        base_path = './img/real'
    else:
        base_path = './img/fake'
    
    dataset_path = []
    
    for i in range(len(dataset)):
        save_path =  f'{base_path}/image_{i}.png'
        dataset_path.append(save_path)
        vutils.save_image(dataset[i], save_path)
    
    return base_path