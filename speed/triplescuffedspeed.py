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

        self.smallest = nn.Conv2d(32, 32, kernel_size=3, dilation=1, stride=2, padding=1, groups=32) #3x3x32 + 32 = 128
        self.small = nn.Conv2d(32, 32, kernel_size=3, dilation=4, stride=2, padding=4, groups=32) #3x3x32 + 32 = 128
        self.large = nn.Conv2d(32, 32, kernel_size=3, dilation=7,stride=2, padding=7, groups=32) #3x3x32 + 32 = 128
        self.largest = nn.Conv2d(32, 32, kernel_size=3, dilation=10, stride=2, padding=10, groups=32) #3x3x32 + 32 = 128
        self.smash1 = nn.Conv2d(128, 64, kernel_size=1) #128x64 + 64 = 8256
        self.bnrelu_ASPP1 = BNReLU(64) # 2 x 64 = 128

        self.narrowest = nn.Conv2d(64, 64, kernel_size=(1, 5), dilation=1, padding=(0, 2), groups=32) #1x5x32 + 32 = 192
        self.narrow = nn.Conv2d(64, 64, kernel_size=(1, 5), dilation=4, padding=(0, 8), groups=32) #1x5x32 + 32 = 192
        self.wide = nn.Conv2d(64, 64, kernel_size=(1, 5), dilation=7, padding=(0, 14), groups=32) #1x5x32 + 32 = 192
        self.widest = nn.Conv2d(64, 64, kernel_size=(1, 5), dilation=10, padding=(0, 20), groups=32) #1x5x32 + 32 = 192
        self.smash2 = nn.Conv2d(256, 256, kernel_size=1) #256x256 + 256 = 65792
        self.bnrelu_ASPP2 = BNReLU(256) # 2 x 128 = 256

        self.layer3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256) #3x3x256 + 256 = 2560
        self.smash3 = nn.Conv2d(256, 256, kernel_size=1) #256x256 + 256 = 65792
        self.bnrelu3 = BNReLU(256) # 2 x 256 = 512

        self.layer4 = nn.Conv2d(256, 256, kernel_size=3, dilation=3, padding=5, groups=256) #3x3x256 + 256 = 2560
        self.smash4 = nn.Conv2d(256, 256, kernel_size=1) #256x256 + 256 = 65792
        self.bnrelu4 = BNReLU(256) # 2 x 256 = 512

        self.classify = nn.Conv2d(256, 2, kernel_size=1) #256x2 + 2 = 514
        #total parameters: 896 + 64 + 4*128 + 8256 + 128 + 4*192 + 65972 + 256 + 2*(2560 + 65792 + 512) + 514 = 215094

    def forward(self, images):
        size = images.shape[2:]
        x = F.interpolate(images, (512, 512), mode="bilinear", align_corners=False)
        x = self.layer1(x)
        x = self.bnrelu1(x)

        st = self.smallest(x)
        s = self.small(x)
        l = self.large(x)
        lt = self.largest(x)
        x = torch.cat((st, s, l, lt), dim=1)
        x = self.smash1(x)
        x = self.bnrelu_ASPP1(x)

        nt = self.narrowest(x)
        n = self.narrow(x)
        w = self.wide(x)
        wt = self.widest(x)
        x = torch.cat((nt, n, w, wt), dim=1)
        x = self.smash2(x)
        x = self.bnrelu_ASPP2(x)

        x = self.layer3(x)
        x = self.smash3(x)
        x = self.bnrelu3(x)

        x = self.layer4(x)
        x = self.smash4(x)
        x = self.bnrelu4(x)

        x = self.classify(x)
        x = F.interpolate(x, size, mode="bilinear", align_corners=False)
        return x
