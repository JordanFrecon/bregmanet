"""
CIFAR10 implementation of BregmanResNet

Based on the ResNet implementation of Yerlan Idelbayev
https://github.com/akamaster/pytorch_resnet_cifar10

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import bregmanet.utils.activation as act

__all__ = ['BregmanResNet', 'bresnet20', 'bresnet32', 'bresnet44', 'bresnet56', 'bresnet110', 'bresnet1202']


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambda_val):
        super(LambdaLayer, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return self.lambda_val(x)


class ClampingLayer(nn.Module):
    def __init__(self, clamp_min=0, clamp_max=1):
        super(ClampingLayer, self).__init__()
        self.min = clamp_min
        self.max = clamp_max

    def forward(self, x):
        return torch.clamp(x, min=self.min+1e-5, max=self.max-1e-5)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BregmanBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='E', activation='atan', batch_norm=True):
        super(BregmanBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if batch_norm else nn.Identity()
        self.activation, self.inverse, self.range = act.get(activation, version='bregman')

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
            elif option == 'C':
                self.shortcut = nn.Sequential(
                    LambdaLayer(lambda x:
                                F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0)),
                    LambdaLayer(lambda x: self.activation(x)),
                    LambdaLayer(lambda x: self.inverse(x))
                )
            elif option == 'D':
                self.shortcut = nn.Sequential(
                    LambdaLayer(lambda x:
                                F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
                )
            elif option == 'E':
                " The padding value is such that, after the inverse activation, it is 0 "
                val = float(self.activation(torch.zeros(1)))
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                            "constant", val))

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = torch.clamp(self.shortcut(x), min=self.range[0]+1e-5, max=self.range[1]-1e-5)
        out += self.inverse(shortcut)
        out = self.activation(out)
        return out


class BisBregmanBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='E', activation='atan', batch_norm=True):
        super(BisBregmanBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if batch_norm else nn.Identity()
        self.activation, self.inverse, self.range = act.get(activation, version='bregman')

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
            elif option == 'C':
                self.shortcut = nn.Sequential(
                    LambdaLayer(lambda x:
                                F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0)),
                    LambdaLayer(lambda x: self.activation(x)),
                    LambdaLayer(lambda x: self.inverse(x))
                )
            elif option == 'D':
                self.shortcut = nn.Sequential(
                    LambdaLayer(lambda x:
                                F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
                )
            elif option == 'E':
                " The padding value is such that, after the inverse activation, it is 0 "
                val = float(self.activation(torch.zeros(1)))
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                            "constant", val))

    def forward(self, x):
        shortcut = torch.clamp(self.shortcut(x), min=self.range[0] + 1e-5, max=self.range[1] - 1e-5)
        out = self.activation(self.inverse(shortcut) + self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = torch.clamp(self.shortcut(out), min=self.range[0]+1e-5, max=self.range[1]-1e-5)
        out += self.inverse(shortcut)
        out = self.activation(out)
        return out


class BregmanResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation='relu', version='standard', batch_norm=True):
        super(BregmanResNet, self).__init__()
        self.in_planes = 16
        self.activation, self.inverse, self.range = act.get(activation, version=version)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16) if batch_norm else nn.Identity()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, activation=activation)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, activation=activation)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, activation=activation)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, activation):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride, activation=activation))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        option = 'A'
        if option == 'A':
            out = self.activation(self.bn1(self.conv1(x)))
        elif option == 'B':
            out = F.relu(self.bn1(self.conv1(x)))
            out = torch.clamp(out, min=self.range[0] + 1e-5, max=self.range[1] - 1e-5)
        else:
            out = x
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def bresnet18(version='bregman', activation='relu', num_classes=10):
    if version.lower() == 'standard':
        return BregmanResNet(BasicBlock, [2, 2, 2, 2], activation=activation, version='standard',
                             num_classes=num_classes)
    elif version.lower() == 'bregman_bis':
        return BregmanResNet(BisBregmanBasicBlock, [2, 2, 2, 2], activation=activation, version='bregman',
                             num_classes=num_classes)
    else:
        return BregmanResNet(BregmanBasicBlock, [2, 2, 2, 2], activation=activation, version='bregman',
                             num_classes=num_classes)


def bresnet20(version='bregman', activation='relu', num_classes=10):
    if version.lower() == 'standard':
        return BregmanResNet(BasicBlock, [3, 3, 3], activation=activation, version='standard',
                             num_classes=num_classes)
    elif version.lower() == 'bregman_bis':
        return BregmanResNet(BisBregmanBasicBlock, [3, 3, 3], activation=activation, version='bregman',
                             num_classes=num_classes)
    else:
        return BregmanResNet(BregmanBasicBlock, [3, 3, 3], activation=activation, version='bregman',
                             num_classes=num_classes)


def bresnet32(version='bregman', activation='relu', num_classes=10):
    if version.lower() == 'standard':
        return BregmanResNet(BasicBlock, [5, 5, 5], activation=activation, version='standard',
                             num_classes=num_classes)
    elif version.lower() == 'bregman_bis':
        return BregmanResNet(BisBregmanBasicBlock, [5, 5, 5], activation=activation, version='bregman',
                             num_classes=num_classes)
    else:
        return BregmanResNet(BregmanBasicBlock, [5, 5, 5], activation=activation, version='bregman',
                             num_classes=num_classes)


def bresnet44(version='bregman', activation='relu', num_classes=10):
    if version.lower() == 'standard':
        return BregmanResNet(BasicBlock, [7, 7, 7], activation=activation, version='standard',
                             num_classes=num_classes)
    elif version.lower() == 'bregman_bis':
        return BregmanResNet(BisBregmanBasicBlock, [7, 7, 7], activation=activation, version='bregman',
                             num_classes=num_classes)
    else:
        return BregmanResNet(BregmanBasicBlock, [7, 7, 7], activation=activation, version='bregman',
                             num_classes=num_classes)


def bresnet56(version='bregman', activation='relu', num_classes=10):
    if version.lower() == 'standard':
        return BregmanResNet(BasicBlock, [9, 9, 9], activation=activation, version='standard',
                             num_classes=num_classes)
    elif version.lower() == 'bregman_bis':
        return BregmanResNet(BisBregmanBasicBlock, [9, 9, 9], activation=activation, version='bregman',
                             num_classes=num_classes)
    else:
        return BregmanResNet(BregmanBasicBlock, [9, 9, 9], activation=activation, version='bregman',
                             num_classes=num_classes)


def bresnet110(version='bregman', activation='relu', num_classes=10):
    if version.lower() == 'standard':
        return BregmanResNet(BasicBlock, [18, 18, 18], activation=activation, version=version,
                             num_classes=num_classes)
    elif version.lower() == 'bregman_bis':
        return BregmanResNet(BisBregmanBasicBlock, [18, 18, 18], activation=activation, version='bregman',
                             num_classes=num_classes)
    else:
        return BregmanResNet(BregmanBasicBlock, [18, 18, 18], activation=activation, version='bregman',
                             num_classes=num_classes)


def bresnet1202(version='bregman', activation='relu', num_classes=10):
    if version.lower() == 'standard':
        return BregmanResNet(BasicBlock, [200, 200, 200], activation=activation, version=version,
                             num_classes=num_classes)
    elif version.lower() == 'bregman_bis':
        return BregmanResNet(BisBregmanBasicBlock, [200, 200, 200], activation=activation, version='bregman',
                             num_classes=num_classes)
    else:
        return BregmanResNet(BregmanBasicBlock, [200, 200, 200], activation=activation, version='bregman',
                             num_classes=num_classes)
