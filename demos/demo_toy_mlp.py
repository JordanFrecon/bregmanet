import torch
from matplotlib import pyplot
import bregmanet as bnn
from demos.utils import optimization as optim, display as dsp, data as data

"""
Training and layer-wise analysis of (Bregman) MLP on the Two-Spiral toy dataset
"""

# CUDA for PyTorch
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    # 2D Toy datasets: Spiral() or Circle()
    value_range = [[-1.57, 1.57], [-1.57, 1.57]]
    data_train = data.Spiral(value_range=value_range)
    data_test = data.Spiral(500, value_range=value_range)
    dataloader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=16, pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=16, pin_memory=True)

    # Define the model
    model_toy = bnn.MLP(version='bregman', activation='atan', hidden_dim=[2] * 3,
                        input_dim=data_train.dim, output_dim=len(data_train.labels), init='rand')

    # Train the model
    print(f"Training of a {model_toy.version} MLP model with {model_toy.num_layers} hidden layers"
          f" made of {model_toy.num_neurons} neurons")
    model_learned, optim_meter = optim.fit(model_toy, data=dataloader_train, lr=1e-2, num_epochs=500, device=device)

    # Display the dataset
    dsp.display_data(data_train, num_fig=1)

    # Display the loss
    pyplot.figure(2)
    pyplot.plot(optim_meter.train_loss, c='k')
    pyplot.xlabel('Epochs')
    pyplot.ylabel('Loss')
    pyplot.show()

    # Display accuracy
    train_accuracy = optim.accuracy(model_learned, dataloader_train, device=device)
    test_accuracy = optim.accuracy(model_learned, dataloader_test, device=device)
    print(f"Training accuracy: {train_accuracy*100:.2f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}")

    # Display layer-wise outputs
    dsp.display_layers_output(model_learned, data_train)
