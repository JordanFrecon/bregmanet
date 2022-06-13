import torch
from matplotlib import pyplot
import bregmanet as bnn
from misc import optimization as optim, display as dsp, utils as utils
import time
import torchvision
import torchvision.transforms as transforms


start_time = time.time()

# CUDA for PyTorch
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    torch.manual_seed(seed=0)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(utils.affine_transformation)
                                    ])

    # Parameters
    batch_size = 16

    # Get data set
    train_set_raw = torchvision.datasets.MNIST(root='../data', train=True, transform=transform, download=True)
    train_set, val_set = torch.utils.data.random_split(train_set_raw, [50000, 10000])
    test_set = torchvision.datasets.MNIST(root='../data', train=False, transform=transform, download=True)

    # Dta loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, pin_memory=True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, pin_memory=True, num_workers=4)

    # Define and fit the model
    model_toy = bnn.MLP(version='bregman', activation='sigmoid', hidden_dim=[784, 2, 2],
                        input_dim=784, output_dim=10, init='zero')

    model_learned, optim_meter = optim.fit(model_toy, data=train_loader, lr=1e-1, num_epochs=100, device=device,
                                           early_stopping=True, data_val=validation_loader)

    # Print performance
    accuracy = optim.accuracy(model_learned, test_loader=test_loader, device=device)
    print('accuracy:\t', accuracy)

    # Display layers output (work best if 2 neurons per layer)
    fig = dsp.display_output(model_learned, train_loader, xlim=[0, 1], ylim=[0, 1])

    # Display loss
    pyplot.figure(2)
    pyplot.plot(optim_meter.train_loss)

    # Display training accuracy
    pyplot.figure(3)
    pyplot.plot(optim_meter.train_accuracy)