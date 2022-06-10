import torch
import torch.nn as nn


""" This module contains various classes and functions to initialize the weights of neural nets """


class LinearSymmetric(nn.Module):
    __constants__ = ['param']

    def __init__(self, param):
        super(LinearSymmetric, self).__init__()
        self.weight_raw = nn.Parameter(torch.zeros(param, param))
        self.bias = nn.Parameter(torch.zeros(param))
        self.weight = .5 * (self.weight_raw.triu() + self.weight_raw.triu().T)

    def forward(self, x0):
        self.weight = .5 * (self.weight_raw.triu() + self.weight_raw.triu().T)
        return torch.matmul(self.weight, x0) + self.bias


def linear_with_init(num_in, num_out, init='rand', weight_norm=False):

    if init == 'zero':
        lin_tmp = nn.Linear(num_in, num_out)
        with torch.no_grad():
            lin_tmp.weight = torch.nn.Parameter(torch.zeros_like(lin_tmp.weight))
            lin_tmp.bias = torch.nn.Parameter(torch.zeros_like(lin_tmp.bias))
    elif init == 'uniform':
        lin_tmp = nn.Linear(num_in, num_out)
        with torch.no_grad():
            lin_tmp.weight = torch.nn.Parameter(torch.ones_like(lin_tmp.weight) / num_in)
            lin_tmp.bias = torch.nn.Parameter(torch.ones_like(lin_tmp.bias) / num_in)
    elif init == 'identity':
        lin_tmp = nn.Linear(num_in, num_out)
        with torch.no_grad():
            lin_tmp.weight = torch.nn.Parameter(torch.eye(num_out, num_in))
            lin_tmp.bias = torch.nn.Parameter(torch.zeros_like(lin_tmp.bias))
    elif init == 'simplex':
        lin_tmp = nn.Linear(num_in, num_out, bias=False)
        with torch.no_grad():
            lin_tmp.weight = torch.nn.Parameter(projection_simplex(lin_tmp.weight))
    else:
        lin_tmp = nn.Linear(num_in, num_out)

    return torch.nn.utils.weight_norm(lin_tmp, name='weight') if weight_norm else lin_tmp


def parameter_initialization(version, init_type):
    if init_type == 'random' or init_type == 'rand':
        return {'hidden': 'rand', 'output': 'rand'}
    elif init_type == 'deterministic':
        if version == 'standard':
            return {'hidden': 'identity', 'output': 'uniform'}
        elif version == 'bregman':
            return {'hidden': 'zero', 'output': 'uniform'}
    else:
        return {'hidden': 'zero', 'output': 'identity'}


def constraint(module):
    if module._get_name() == 'Linear':
        with torch.no_grad():
            module.weight = torch.nn.Parameter(projection_simplex(module.weight))
    return module
    # """ Clamp the weights norm to 1"""
    # if self.weight_norm:
    #    with torch.no_grad():
    #        for ll in range(self.num_layers):
    #            self.lin[ll].weight_g.dataset = torch.clamp_max(self.lin[ll].weight_g, max=1)


def projection_simplex(v, radius=1):
    """
    Pytorch implementation (maybe not optimal) of the projection into the simplex.
    """
    n_feat = v.shape[1]
    n_neuron = v.shape[0]
    u, _ = torch.sort(v)
    cssv = torch.cumsum(u, dim=1) - radius
    ind = torch.arange(n_feat, device=v.device).repeat(n_neuron, 1) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond].reshape(n_neuron, n_feat)[:, -1]
    theta = torch.div(cssv[cond].reshape(n_neuron, n_feat)[:, -1], rho)
    relu = torch.nn.ReLU()
    return relu(v - theta.reshape(n_neuron, 1))
