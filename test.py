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
import fid_score
from fid_score.fid_score import FidScore

from datasets import *
from models import *
from utils import *

CKPT_NETD= 'netD_epoch_{}.pth'
CKPT_NETG = 'netG_epoch_{}.pth'

def test(epoch, num_test=1000, ckpt_dir='./checkpoint/'):
    # load trained model
    G = Generator()
    G.cuda()
    checkpointG = torch.load(ckpt_dir + CKPT_NETG.format(epoch))
    G.load_state_dict(checkpointG['model_state_dict'])
    G.eval()

    # generate
    with torch.no_grad():
        noise = torch.randn(num_test, 100).view(-1, 100, 1, 1).cuda()
        fake = G(noise)
    
    fake_img = (fake * 0.5) + 0.5
    fake_list = save_image_list(fake_img, False)

    # load real images
    dataloader = init_data(batch_size=1000)
    for i, (data, _) in enumerate(dataloader):
        real_dataset = data
        break
    real_list = save_image_list(real_dataset, True)

    fid = FidScore([real_list, fake_list], 'cuda', 1000)
    fid_value = fid.calculate_fid_score()
    print (f'FID score: {fid_value}')


if __name__ == '__main__':
    test(19)