import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                self.conv2,
                self.bn2,
                nn.ReLU()
                
            )

    def forward(self, x):
        out1 = F.relu(self.bn1(self.maxpool1(self.conv1(x))))
        out = F.relu(self.bn2(self.conv2(out1)))
        out = self.shortcut(out)
        # F.relu(self.bn2(self.conv2(out)))
        out += out1
        out = F.relu(out)
        return out

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, 1)
        self.max_pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        out = F.relu(self.bn1(self.max_pool(self.conv1(x))))
        return out



class ResNetCustom(nn.Module):
    def __init__(self, block, block2, num_blocks, num_classes=10):
        super(ResNetCustom, self).__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # self._prep_layer = F.relu(self.bn1(self.conv1(x)))

        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)

        self.layer2 = self._make_layer2(block2, 256, num_blocks[1], stride=1)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=3, bias=False)
        self.max_pool = nn.MaxPool2d(4,2)
        self.bn2 = nn.BatchNorm2d(256)

        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=1)

        self.max_pool2 = nn.MaxPool2d(4)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _make_layer2(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        r1 = self.layer1(x)
        out = self.layer2(r1)
        r2 = self.layer3(out)
        out = self.max_pool2(r2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def CustomResNet():
    return ResNetCustom(BasicBlock,BasicBlock2, [1, 1, 1],)


def test():
    net = CustomResNet()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())