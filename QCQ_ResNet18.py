import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=2):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)  # ! (h-3+2)/2 + 1 = h/2 图像尺寸减半
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)  # ! h-3+2*1+1=h 图像尺寸没变化
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),  # ! 这句话是针对原图像尺寸写的，要进行element wise add
            # ! 因此图像尺寸也必须减半，(h-1)/2+1=h/2 图像尺寸减半
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        out = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        # shortcut
        # ! element wise add [b,ch_in,h,w] [b,ch_out,h,w] 必须当ch_in = ch_out时才能进行相加
        out = x + self.extra(out)  # todo self.extra强制把输出通道变成一致
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # ! 图像尺寸不变
            nn.BatchNorm2d(64)
        )
        # 4个ResBlock
        #  [b,64,h,w] --> [b,128,h,w]
        self.block1 = ResBlock(64, 128)
        #  [b,128,h,w] --> [b,256,h,w]
        self.block2 = ResBlock(128, 256)
        #  [b,256,h,w] --> [b,512,h,w]
        self.block3 = ResBlock(256, 512)
        #  [b,512,h,w] --> [b,512,h,w]
        self.block4 = ResBlock(512, 512)

        self.outlayer = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # [b,64,h,w] --> [b,1024,h,w]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # print("after conv:",x.shape)
        # [b,512,h,w] --> [b,512,1,1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # flatten
        x = x.view(x.shape[0], -1)
        x = self.outlayer(x)
        return x
