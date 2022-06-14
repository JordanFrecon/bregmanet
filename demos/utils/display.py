import torch
from matplotlib import pyplot
from demos.utils.utils import get_nn_parameters
import numpy as np


def display_data(data, num_fig=None, cmap='viridis', filename=None, xlim=None, ylim=None):
    """Display the dataset (either 1d or 2d)"""

    if data.dim == 2:
        xx = [[], []]
        yy = []
        for xv, yv in data:
            xx[0].append(xv[0].item())
            xx[1].append(xv[1].item())
            yy.append(yv.item())

        pyplot.figure(num_fig)
        pyplot.scatter(xx[0], xx[1], c=yy, cmap=cmap)
        pyplot.xlabel('Input 1')
        pyplot.ylabel('Input 2')
        pyplot.title('Training dataset')

    else:
        pyplot.figure(num_fig)
        for xv, yv in data:
            pyplot.plot(xv, yv, 'x', c='k')
            pyplot.xlabel('Input')
            pyplot.ylabel('Target')
            pyplot.title('Training dataset')

    if xlim is not None:
        pyplot.xlim(xlim)
    if ylim is not None:
        pyplot.ylim(ylim)
    if filename is not None:
        pyplot.rcParams["font.family"] = "serif"
        pyplot.rcParams["font.size"] = "20"
        pyplot.xticks([-1, 1])
        pyplot.yticks([-1, 1])
        pyplot.savefig(filename + "_dataset.pdf", bbox_inches="tight")


def get_vector_field(model, num_bins=10, input_range=None):
    """Get the vector field at each layer from the NN weights and biases"""
    # Get FNN parameters
    model = model.to('cpu')
    weights, biases = get_nn_parameters(model)

    # Get vector field
    if len(input_range) == 2:
        x1_range, x2_range = input_range
        x1, x2 = np.meshgrid(np.linspace(x1_range[0], x1_range[1], num_bins),
                             np.linspace(x2_range[0], x2_range[1], num_bins))
        gx1_all = []
        gx2_all = []
        for id_layer in range(model.num_layers):
            gx1 = np.empty_like(x1)
            gx2 = np.empty_like(x2)
            for ii in range(num_bins):
                for jj in range(num_bins):
                    c = (-(np.matmul(weights[id_layer], [x1[ii][jj], x2[ii][jj]]) + (biases[id_layer]))).tolist()
                    gx1[ii][jj] = c[0]
                    gx2[ii][jj] = c[1]
            gx1_all.append(gx1)
            gx2_all.append(gx2)

        return {'input': [x1, x2], 'gradient': [gx1_all, gx2_all]}

    elif len(input_range) == 1:
        [x1_range] = input_range
        x1 = np.linspace(x1_range[0], x1_range[1], num_bins)
        gx1_all = []
        for id_layer in range(model.num_layers):
            gx1 = np.empty_like(x1)
            for ii in range(num_bins):
                gx1[ii] = - (weights[id_layer][0][0] * x1[ii] + biases[id_layer])
            gx1_all.append(gx1)

        return {'input': [x1], 'gradient': [gx1_all]}


def integrate_vector_field(vec_field):
    """Estimate the mapping function at each layer by discrete integration of its gradient vector fields"""

    input_discrete, grad_discrete = vec_field['input'], vec_field['gradient']

    if len(input_discrete) == 1:
        [x_bin], [gx_bin] = input_discrete, grad_discrete

        f_all = np.empty_like(gx_bin)
        for idl in range(len(gx_bin)):  # i.e., the number of layers
            f_val = 0
            f = np.empty_like(x_bin)
            for idg, g_val in enumerate(gx_bin[idl]):
                f[idg] = f_val
                if idg < len(x_bin) - 1:
                    f_val = f_val + g_val * (x_bin[idg + 1] - x_bin[idg])
            f_all[idl] = f

        return {'input': [x_bin], 'function': [f_all]}

    else:
        return None


def integrate_vector_field_exp(vec_field):
    """Experimental"""

    input_discrete, grad_discrete = vec_field['input'], vec_field['gradient']

    if len(input_discrete) == 1:
        [x_bin], [gx_bin] = input_discrete, grad_discrete

        # Fist pass through the last layer
        f_val = 0
        f = np.empty_like(x_bin)
        for idg, g_val in enumerate(gx_bin[len(gx_bin) - 1]):
            f[idg] = f_val
            if idg < len(x_bin) - 1:
                f_val = f_val + g_val * (x_bin[idg + 1] - x_bin[idg])
        f0 = f

        # And then ..
        f_all = np.empty_like(gx_bin)
        f_all[len(gx_bin) - 1] = f0
        for idl in reversed(range(len(gx_bin) - 1)):
            f = np.empty_like(x_bin)
            for idg, g_val in enumerate(gx_bin[idl]):
                if idg < len(x_bin) - 1:
                    f[idg] = f_all[idl + 1][idg] + g_val * (x_bin[idg + 1] - x_bin[idg])
            f_all[idl] = f

        return {'input': [x_bin], 'function': [f_all]}

    else:
        return None


def integrate_vector_field_cumsum(vec_field):
    """Experimental"""

    input_discrete, grad_discrete = vec_field['input'], vec_field['gradient']

    if len(input_discrete) == 1:
        [x_bin], [gx_bin] = input_discrete, grad_discrete

        f_val = 0
        f = np.empty_like(x_bin)
        for idg in range(len(x_bin)):
            for idl in range(len(gx_bin)):  # i.e., the number of layers
                g_val = gx_bin[idl][idg]
                f[idg] = f_val
                if idg < len(x_bin) - 1:
                    f_val = f_val + g_val * (x_bin[idg + 1] - x_bin[idg])

        return {'input': [x_bin], 'function': [f]}

    else:
        return None


def display_layer_function(function, num_fig=None, legend=None):
    if function is not None:

        input_discrete, output_discrete = [function.get(b) for b in function.keys()]

        if len(input_discrete) == 2:
            [x_bin, y_bin], [fx_bin, fy_bin] = input_discrete, output_discrete

            for idl in range(len(fx_bin)):  # i.e., the number of layers
                pyplot.figure(num_fig + idl)
                pyplot.quiver(x_bin, y_bin, fx_bin[idl], fy_bin[idl], np.sqrt(fx_bin[idl] ** 2 + fy_bin[idl] ** 2))
                pyplot.title(r"$%s_{%s}$" % (legend, idl + 1))
                pyplot.xlabel('Input 1')
                pyplot.ylabel('Input 2')
                pyplot.show()

        else:
            [x_bin], [fx_bin] = input_discrete, output_discrete

            pyplot.figure(num_fig)
            for idl in range(len(fx_bin)):  # i.e., the number of layers
                if legend is not None:
                    pyplot.plot(x_bin, fx_bin[idl], label=r"$%s_{%s}$" % (legend, idl + 1))
                else:
                    pyplot.plot(x_bin, fx_bin[idl])
            pyplot.xlabel('Input')
            pyplot.legend()


def display_layers_output(model, data, num_fig=None, xlim=None, ylim=None, filename=None):
    model = model.to('cpu')
    x_all = np.empty([model.num_layers, data.__len__(), data.dim])
    y_all = np.empty(data.__len__())

    for ids, [x, y] in enumerate(data):
        for idl in range(model.num_layers):
            x = model.activation(model.offset(x.data) + model.lin[idl](x.cpu()))
            x_all[idl, ids, :] = np.array(x.cpu().detach().numpy())

        y_all[ids] = y

    for idl in range(model.num_layers):
        pyplot.figure(num_fig + idl) if num_fig is not None else pyplot.figure()
        pyplot.scatter(x_all[idl][:, 0], x_all[idl][:, 1], c=y_all)
        pyplot.title(r"Output of layer $%s$" % (idl + 1))
        pyplot.xlabel('Input 1')
        pyplot.ylabel('Input 2')
        if xlim is not None:
            pyplot.xlim(xlim)
        if ylim is not None:
            pyplot.ylim(ylim)
        # pyplot.xlim(dataset.range[0])
        # pyplot.ylim(dataset.range[1])
        pyplot.show()

        if filename is not None:
            pyplot.rcParams["font.family"] = "serif"
            pyplot.rcParams["font.size"] = "20"
            pyplot.savefig(filename + "_layer" + str(idl) + ".pdf", bbox_inches="tight")

    if data.dim == 2:
        if xlim is None:
            x_bin = np.linspace(model.range[0] - 1e-5, model.range[1] + 1e-5, 100)
        else:
            x_bin = np.linspace(xlim[0], xlim[1], 100)
        if len(model.output.bias) == 1:
            y_sep = (.5 - model.output.bias.item() - model.output.weight[0][0].item() * x_bin) / \
                    model.output.weight[0][1].item()
            pyplot.plot(x_bin, y_sep, 'k')
        else:
            y_sep = (model.output.bias[1].item() - model.output.bias[0].item() - (
                    model.output.weight[0][0].item() - model.output.weight[1][0].item()) * x_bin) / \
                    (model.output.weight[0][1].item() - model.output.weight[1][1].item())
            pyplot.plot(x_bin, y_sep, 'k')
        if xlim is not None:
            pyplot.xlim(xlim)
        if ylim is not None:
            pyplot.ylim(ylim)
        # pyplot.xlim(dataset.range[0])
        # pyplot.ylim(dataset.range[1])


def display_output(model, data, num_fig=None, xlim=None, ylim=None):
    model = model.to('cpu')
    x_all = np.empty([data.__len__() * data.batch_size, 2])
    y_all = np.empty(data.__len__() * data.batch_size)

    for ids, (x, y) in enumerate(data):
        for idl in range(model.num_layers):
            if model.version == 'bregman':
                x = model.activation(model.offset(model.reparametrization[idl](x)) + model.lin[idl](x))
            else:
                x = model.activation(model.lin[idl](x))

        vec = np.arange(ids * data.batch_size, (ids + 1) * data.batch_size)
        x_all[vec] = np.array(x.cpu().detach().numpy())
        y_all[vec] = y

    fig = pyplot.figure(num_fig) if num_fig is not None else pyplot.figure()
    pyplot.set_cmap('tab10')
    pyplot.scatter(x_all[:, 0], x_all[:, 1], c=y_all)
    pyplot.title(r"Final layer")
    pyplot.xlabel('Output 1')
    pyplot.ylabel('Output 2')
    if xlim is not None:
        pyplot.xlim(xlim)
    if ylim is not None:
        pyplot.ylim(ylim)
    return fig
    # pyplot.xlim(dataset.range[0])
    # pyplot.ylim(dataset.range[1])


def display_grid_layers_output(model, data, num_fig=None, xlim=None, nbins=1000):
    x1_all_tmp = np.linspace(data.range[0][0], data.range[0][1], nbins)
    x2_all_tmp = np.linspace(data.range[1][0], data.range[1][1], nbins)
    x_all = []
    for ii in x1_all_tmp:
        for jj in x2_all_tmp:
            x_all.append([ii, jj])
    x = torch.tensor(x_all).type(torch.float)
    xinit = x

    model = model.to('cpu')
    x_all = np.empty([model.num_layers, nbins ** 2, data.dim])

    for idl in range(model.num_layers):
        x = model.activation(model.offset(x.data) + model.lin[idl](x.cpu()))
        x_all[idl, :, :] = np.array(x.cpu().detach().numpy())

    # Get boundary
    x_bin = np.linspace(model.range[0] - 1e-5, model.range[1] + 1e-5, nbins)
    y_sep = (model.output.bias[1].item() - model.output.bias[0].item() - (
            model.output.weight[0][0].item() - model.output.weight[1][0].item()) * x_bin) / \
            (model.output.weight[0][1].item() - model.output.weight[1][1].item())
    xlast = np.stack([x_bin, y_sep]).transpose()

    print(x_all.shape)
    xoutput = torch.tensor(x_all[-1])
    xbnd = []
    ybnd = []
    for xval in xlast:
        ind = torch.sqrt(torch.sum( (xoutput - torch.tensor(xval))**2, 1)).argmin(dim=0)
        xbnd.append(xinit[ind][0].item())
        ybnd.append(xinit[ind][1].item())

    return xbnd, ybnd


def display_grid_layers_output_v2(model, data, num_fig=None, xlim=None, nbins=1000, eps=1e-1):
    x1_all_tmp = np.linspace(data.range[0][0], data.range[0][1], nbins)
    x2_all_tmp = np.linspace(data.range[1][0], data.range[1][1], nbins)
    x_all = []
    for ii in x1_all_tmp:
        for jj in x2_all_tmp:
            x_all.append([ii, jj])
    x = torch.tensor(x_all).type(torch.float)
    xinit = x

    model = model.to('cpu')
    x_all = np.empty([model.num_layers, nbins ** 2, data.dim])

    for idl in range(model.num_layers):
        x = model.activation(model.offset(x.data) + model.lin[idl](x.cpu()))
        x_all[idl, :, :] = np.array(x.cpu().detach().numpy())

    x_end = model.output(x)

    xbnd = []
    ybnd = []
    for ind, xval in enumerate(x_end):
        if abs(xval[0]-xval[1])/abs(xval[0]+xval[1]) < eps:
            xbnd.append(xinit[ind][0].item())
            ybnd.append(xinit[ind][1].item())

    return xbnd, ybnd