import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
# import 

from datasets import *
from models import *
from utils import *

def train():
    G = Generator()
    D = Discriminator()
    optG = optim.Adam(G.parameters(), lr=0.0002)
    optD = optim.Adam(D.parameters(), lr=0.0002)
    dataloader = init_data()
    G.cuda()
    D.cuda()

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(128, 100).view(-1, 100, 1, 1).cuda()
    training_progress_images_list = []
    for epoch in range(200):
        
        for batch in tqdm(dataloader, leave=False, ncols=70):
            # Upate D network
            # train with real
            # print(batch[0].shape, batch[1].shape)
            D.zero_grad()
            batch = batch[0].cuda()
            b = batch.shape[0]
            label_real = torch.ones((b, )).cuda()
            out = D(batch)
            errD_real = criterion(out, label_real)
            D_x = out.mean().item()

            # train with fake
            noise = torch.randn(b, 100).view(-1, 100, 1, 1).cuda()
            fake = G(noise)
            label_fake = torch.zeros((b, )).cuda()
            out = D(fake)
            errD_fake = criterion(out, label_fake)
            D_G_z1 = out.mean().item()

            # loss backward
            errD = errD_real + errD_fake
            errD.backward(retain_graph=True)
            optD.step()

            # UPDATE G  
            G.zero_grad()
            out = D(fake)
            errG = criterion(out, label_real)
            D_G_z2 = out.mean().item()

            errG.backward()
            optG.step()

        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, 200, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        fake = G(fixed_noise)
        training_progress_images_list = save_gif(training_progress_images_list, (fake * 0.5) + 0.5)  # Save fake image while training!
        torch.save({
                'epoch': epoch,
                'model_state_dict': G.state_dict(),
                'optimizer_state_dict': optG.state_dict(),
                }, './checkpoint/netG_epoch_%d.pth' % (epoch))
        torch.save({
                'epoch': epoch,
                'model_state_dict': D.state_dict(),
                'optimizer_state_dict': optD.state_dict(),
                }, './checkpoint/netD_epoch_%d.pth' % (epoch))


if __name__ == '__main__':
    train()