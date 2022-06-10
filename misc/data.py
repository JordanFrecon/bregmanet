import torch
import numpy as np
import torch.utils.data as data
import math


# Remark: if dataset is made in numpy then dtype will be double


class Hill(data.Dataset):
    """Toy dataset set where x ~ ]0,1[ and y=0 if x<.5 and 1 otherwise"""

    def __init__(self, num_samples=1000, eps=1e-5, setting='classification'):
        self.num_samples = num_samples
        self.dim = 1
        self.range = [[0, 1]]
        self.labels = [0, 1]
        self.setting = setting

        self.x = eps + (1. - 2 * eps) * torch.rand(num_samples, 1)
        self.y = 0. * (self.x < .5) + 1. * (self.x >= .5)
        if self.setting == 'classification':
            self.y = self.y.flatten().long()

    def __getitem__(self, item):
        if self.setting == 'classification':
            return self.x[item], self.y[item]
        else:
            return self.x[np.newaxis, item], self.y[np.newaxis, item]

    def __len__(self):
        return self.num_samples


class TwinHills(data.Dataset):

    def __init__(self, num_samples=1000, eps=1e-5, setting='classification'):
        self.num_samples = num_samples
        self.dim = 1
        self.range = [[0, 1]]
        self.labels = [0, .5, 1]
        self.setting = setting

        self.x = eps + (1. - 2 * eps) * torch.rand(num_samples, 1)
        self.y = 0. * (self.x < 1 / 3) + 1. * (self.x > 2 / 3) + .5 * (self.x >= 1 / 3) * (self.x <= 2 / 3)
        if self.setting == 'classification':
            self.y[self.y == 1] = 2
            self.y[self.y == .5] = 1
            self.y = self.y.flatten().long()

    def __getitem__(self, item):
        if self.setting == 'classification':
            return self.x[item], self.y[item]
        else:
            return self.x[np.newaxis, item], self.y[np.newaxis, item]

    def __len__(self):
        return self.num_samples


class Spiral(data.Dataset):

    def __init__(self, num_samples=1000, setting='classification', value_range=None):
        n = int(num_samples / 2)
        self.num_samples = num_samples
        self.dim = 2
        self.range = [[-1, 1], [-1, 1]] if value_range is None else value_range
        self.labels = [0, 1]

        theta = torch.sqrt(torch.rand(n)) * 2 * math.pi

        r_a = 2 * theta + math.pi
        data_a = torch.stack((torch.cos(theta) * r_a, torch.sin(theta) * r_a), 1)
        x_a = data_a + torch.randn(n, 2)

        r_b = -2 * theta - math.pi
        data_b = torch.stack((torch.cos(theta) * r_b, torch.sin(theta) * r_b), 1)
        x_b = data_b + torch.randn(n, 2)

        x = torch.cat((x_a, x_b), dim=0)
        y = torch.cat((torch.zeros((n, 1)), torch.ones((n, 1))), dim=0)
        perm = torch.randperm(num_samples)
        self.x = x[perm] / (x.abs().max().ceil())

        for ii in range(2):
            for nn in range(num_samples):
                self.x[nn][ii] = self.range[ii][0] + .5*(self.range[ii][1] - self.range[ii][0])*(self.x[nn][ii]+1)

        if setting == 'classification':
            # Works for BCE loss
            self.y = y[perm].squeeze().long()
        else:
            # Works for MSE loss
            self.y = y[perm]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.num_samples


class Circle(data.Dataset):

    def __init__(self, num_samples=1000, setting='classification'):
        self.num_samples = num_samples
        self.dim = 2
        self.range = [[-1, 1], [-1, 1]]
        self.labels = [0, 1]

        angle = torch.rand(num_samples)
        sign = torch.sign(torch.randn(num_samples))
        radius = .2 * torch.rand(num_samples) + .1 * (sign == 1) + .7 * (sign == -1)

        self.x = torch.empty([2, num_samples])
        self.x[0] = radius * torch.cos(2 * math.pi * angle)
        self.x[1] = radius * torch.sin(2 * math.pi * angle)
        self.y = torch.empty((num_samples, 1))
        self.y[:, 0] = 1. * (sign == 1) + 0. * (sign == -1)

        if setting == 'classification':
            # Works for BCE loss
            self.y = self.y.squeeze().long()

    def __getitem__(self, item):
        return self.x[:, item], self.y[item]

    def __len__(self):
        return self.num_samples
