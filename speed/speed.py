import torch
import torch.nn as nn
import torch.nn.functional as F


def BNReLU(num_features):
    return nn.Sequential(
        nn.BatchNorm2d(num_features),
        nn.ReLU())

class C9H13N(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1) # 3x3x3x32 + 32 = 896
        self.bnrelu1 = BNReLU(32) # 2 x 32 = 64
        self.narrowest = nn.Conv2d(32, 32, kernel_size=(1, 3), dilation=1, padding=(0, 1), groups=32) #1x3x32 + 32 = 96
        self.narrow = nn.Conv2d(32, 32, kernel_size=(1, 3), dilation=4, padding=(0, 4), groups=32) #1x3x32 + 32 = 96
        self.wide = nn.Conv2d(32, 32, kernel_size=(1, 3), dilation=8, padding=(0, 8), groups=32) #1x3x32 + 32 = 96
        self.widest = nn.Conv2d(32, 32, kernel_size=(1,3), dilation=12, padding=(0, 12), groups=32) #1x3x32 + 32 = 96
        self.smash1 = nn.Conv2d(128, 128, kernel_size=1) #128x128 + 128 = 16512
        self.bnrelu_ASPP = BNReLU(128) # 2 x 128 = 256
        self.layer2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128) #3x3x128 + 128 = 1280
        self.smash2 = nn.Conv2d(128, 128, kernel_size=1) #128x128 + 128 = 16512
        self.bnrelu2 = BNReLU(128) # 2 x 128 = 256
        self.layer3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) #3x3x128x256 + 256 = 295168
        self.bnrelu3 = BNReLU(256) # 2 x 256 = 512
        self.classify = nn.Conv2d(256, 2, kernel_size=1) #256x2 + 2 = 514
        #total parameters: 896 + 64 + 4*96 + 16512 + 256 + 1280 + 16512 + 256 + 295168 + 512 + 514 = 332354

    def forward(self, images):
        size = images.shape[2:]
        x = F.interpolate(images, (512, 512), mode="bilinear", align_corners=False)
        x = self.layer1(x)
        x = self.bnrelu1(x)
        nt = self.narrowest(x)
        n = self.narrow(x)
        w = self.wide(x)
        wt = self.widest(w)
        x = torch.cat((nt, n, w, wt), dim=1)
        x = self.smash1(x)
        x = self.bnrelu_ASPP(x)
        x = self.layer2(x)
        x = self.smash2(x)
        x = self.bnrelu2(x)
        x = self.layer3(x)
        x = self.bnrelu3(x)
        x = self.classify(x)
        x = F.interpolate(x, size, mode="bilinear", align_corners=False)
        return x
