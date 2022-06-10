import torch
import numpy as np
from matplotlib import pyplot
import networks as bnn
from networks.utils import optimization as optim, display as dsp, data as data
import time

start_time = time.time()

# CUDA for PyTorch
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    seed = 3
    torch.manual_seed(seed=seed)
    np.random.seed(seed)

    # 1D: Hill() or TwinHills()
    # 2D: Spiral() or Circle()
    value_range = [[-1.57, 1.57], [-1.57, 1.57]]
    data_train = data.Spiral(value_range=value_range)
    data_val = data.Spiral(500)
    data_gen = torch.utils.data.DataLoader(dataset=data_train, batch_size=int(16), pin_memory=True)
    data_valid = torch.utils.data.DataLoader(dataset=data_val, batch_size=int(16), pin_memory=True)

    # Define and fit the model
    model_toy = bnn.MLP(version='bregman', activation_name='scaled_tanh', hidden_dim=[2] * 3,
                        input_dim=data_train.dim, output_dim=len(data_train.labels), init='rand')

    model_learned, optim_meter = optim.fit(model_toy, data=data_gen, lr=1e-3, num_epochs=500, device=device)

    # Display the dataset
    pyplot.rcParams["font.family"] = "serif"
    pyplot.rcParams["font.size"] = "20"
    dsp.display_data(data_train, num_fig=1)  # , xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])

    # Display the loss
    pyplot.figure(2)
    pyplot.plot(optim_meter.train_loss, c='k')
    pyplot.xlabel('Epochs')
    pyplot.ylabel('Loss')
    pyplot.show()

    # Display accuracy
    accuracy = optim.accuracy(model_learned, data_gen, device=device)
    print(accuracy)

    dsp.display_layers_output(model_learned, data_train)


    print("--- %s seconds ---" % (time.time() - start_time))
