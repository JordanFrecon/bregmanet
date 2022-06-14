import torch
import numpy as np


def get_nn_parameters(model):
    """Get the NN weights and biases"""
    weights = []
    biases = []
    for id_layer in range(model.num_layers):
        weights.append(model.lin[id_layer].weight.data.tolist())
        biases.append(model.lin[id_layer].bias.data.tolist())
    return weights, biases


def get_nn_parameters_norm(model, offset=None):
    """ Get the NN weights norms and biases norm at each layer"""
    bias_norm = []
    weight_norm = []
    for name, param in model.named_parameters():
        if 'bias' in name and 'output' not in name:
            bias_norm.append(torch.norm(param))
        elif 'weight' in name and 'output' not in name:
            if offset is None:
                weight_norm.append(torch.norm(param))
            else:
                weight_norm.append(torch.norm(param - offset))
    return weight_norm, bias_norm


def get_nn_outputs_norm(model, data, device=torch.device('cpu')):
    num_layers = len(model.lin)
    output_norms = torch.zeros(num_layers)
    for x, _ in data:
        x_old = x.to(device=device)
        for ind in range(num_layers):
            x_var = model.activation(model.offset(x_old) + model.lin[ind].forward(x_old))
            output_norms[ind] += float(torch.norm(x_var - x_old, 'fro') ** 2)
            x_old = x_var
    return output_norms / data.dataset.__len__()


def get_nn_outputs_norm_stat(model, data, device=torch.device('cpu')):
    num_layers = len(model.lin)
    num_data = data.dataset.__len__()
    output_norms = torch.zeros(num_layers, num_data)
    for ind_data, (x, _) in enumerate(data):
        x_old = x.to(device=device)
        ind_vec = np.arange(ind_data * x.shape[0], (ind_data + 1) * x.shape[0])
        for ind in range(num_layers):
            x_var = model.activation(model.offset(x_old) + model.lin[ind].forward(x_old))
            tmp = float(torch.sum((x_var - x_old) ** 2, dim=[1]))
            output_norms[ind, ind_vec[:]] = tmp
            x_old = x_var
    return output_norms


def affine_transformation(var_in, cmin=0, cmax=1, eps=1e-5):
    #return offset + (1 - 2 * offset) * (np.array(var_in) / 255).flatten()
    return (cmin+eps) + (cmax - cmin - 2 * eps) * var_in.flatten()


def affine_transformation_bis(var_in, cmin=-1, cmax=1, eps=1e-5):
    #return offset + (1 - 2 * offset) * (np.array(var_in) / 255).flatten()
    return (cmin+eps) + (cmax - cmin - 2 * eps) * var_in.flatten()


def affine_transformation_2d(var_in, offset=1e-5):
    """2D rescaling into [0,1] with small offset. Also add a channel for RGB (here no colors)"""
    tmp = offset + (1 - 2 * offset) * (np.array(var_in) / 255)
    return tmp[None, :, :]


def vectorize_label(var_in):
    label_convert = torch.zeros(10)
    label_convert[var_in] = 1
    return label_convert
