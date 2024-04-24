import os
import random
import torch
from torchvision.io import read_image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch.utils.data import Dataset


class KittiTraining(Dataset):
    def __init__(self):
        self.img_dir = 'data/kitti/training/image_2'
        self.label_dir = 'data/kitti/training/gt_image_2'
        self.imgs = os.listdir(self.img_dir)  # assuming that the images and labels have same names in their folders

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.imgs[item])
        label_path = os.path.join(self.label_dir, self.imgs[item])
        img = read_image(img_path)
        label = read_image(label_path)
        shape = [375, 1242]

        if torch.rand(1).item() > 0.5:  # 50% chance of Horizontal flip
            img = TF.hflip(img)
            label = TF.hflip(label)

        if torch.rand(1).item() > 0.5:  # 50 % chance of random resized crop
            hi = max(torch.rand(1).item(), 0.1)
            i, j, h, w = T.RandomResizedCrop.get_params(img, scale=[0.08, hi], ratio=[2, 4])
            img = TF.crop(img, i, j, h, w)
            label = TF.crop(label, i, j, h, w)

        if torch.rand(1).item() > 0.5:  # 50% chance of Gaussian Noise
            mean = 0
            std = 20
            noise = torch.randn(img.shape[1:]) * std + mean
            noise = noise.expand(img.shape)
            noise = torch.round(noise).type(torch.int8)
            img = torch.clamp(img + noise, 0, 255).type(torch.uint8)

        if torch.rand(1).item() > 0.5: # 50% chance of color jitter
            jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            img = jitter(img)

        img = TF.resize(img, shape)
        label = TF.resize(label, shape)

        brightness_factor = random.uniform(0.6, (5/3))
        img = TF.adjust_brightness(img, brightness_factor)
        img = TF.normalize(img.type(torch.float), [0, 0, 0], [1, 1, 1], inplace=True)
        label = label[2, ...] / 255  # not sure what type this should be
        return img, label

# loader = KittiTraining()
# import matplotlib.pyplot as plt
# img, label = loader.__getitem__(30)
# plt.imshow(  img.permute(1, 2, 0)  )
# plt.show()
# plt.imshow(  label  )
# plt.show()

