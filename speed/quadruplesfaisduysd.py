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

        self.layer2 = nn.Conv2d(32, 32, kernel_size=3, dilation=3, padding=3, groups=32) # 3x3x3x32 + 32 = 896
        self.smash0 = nn.Conv2d(32, 32, kernel_size=1)
        self.bnrelu2 = BNReLU(32) # 2 x 32 = 64

        self.small = nn.Conv2d(32, 32, kernel_size=3, dilation=1, stride=2, padding=1, groups=32) #3x3x32 + 32 = 128
        self.medium = nn.Conv2d(32, 32, kernel_size=3, dilation=6,stride=2, padding=6, groups=32) #3x3x32 + 32 = 128
        self.large = nn.Conv2d(32, 32, kernel_size=3, dilation=11, stride=2, padding=11, groups=32) #3x3x32 + 32 = 128
        self.smash1 = nn.Conv2d(96, 64, kernel_size=1) #128x64 + 64 = 8256
        self.bnrelu_ASPP1 = BNReLU(64) # 2 x 64 = 128

        self.narrow = nn.Conv2d(64, 64, kernel_size=(1, 5), dilation=1, padding=(0, 2), groups=64) #1x5x32 + 32 = 192
        self.lukewarm = nn.Conv2d(64, 64, kernel_size=(1, 5), dilation=6, padding=(0, 12), groups=64) #1x5x32 + 32 = 192
        self.wide = nn.Conv2d(64, 64, kernel_size=(1, 5), dilation=11, padding=(0, 22), groups=64) #1x5x32 + 32 = 192
        self.smash2 = nn.Conv2d(192, 128, kernel_size=1) #256x256 + 256 = 65792
        self.bnrelu_ASPP2 = BNReLU(128) # 2 x 128 = 256

        self.layer3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128) #3x3x256 + 256 = 2560
        self.smash3 = nn.Conv2d(128, 256, kernel_size=1) #256x256 + 256 = 65792
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

        x = self.layer2(x)
        x = self.smash0(x)
        x = self.bnrelu2(x)

        s = self.small(x)
        m = self.medium(x)
        l = self.large(x)
        x = torch.cat((s, m, l), dim=1)
        x = self.smash1(x)
        x = self.bnrelu_ASPP1(x)

        n = self.narrow(x)
        lw = self.lukewarm(x)
        w = self.wide(x)
        x = torch.cat((n, lw, w), dim=1)
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
