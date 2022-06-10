import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from misc import optimization as optim
import bregmanet as brn

# In the original paper 'Deep Residual Learning for Image Recognition' they use:
#   - weight_decay=1e-4 and momentum=0.9
#   - learning rate of 0.1, divide it by 10 at 32k and 48k iterations

# CUDA for PyTorch
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    # Dataset: CIFAR10
    dataset = torchvision.datasets.CIFAR100

    # Data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Data splits
    train_set_raw = dataset(root='../data', train=True, download=True, transform=train_transform)
    train_set, val_set = torch.utils.data.random_split(train_set_raw, [45000, 5000])
    test_set = dataset(root='../data', train=False, download=True, transform=test_transform)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=128, pin_memory=True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=128, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=128, pin_memory=True, num_workers=4)

    # Model: BregmanResNet20
    bresnet = brn.bresnet20(activation='softplus', version='bregman', num_classes=100)

    # Training the model
    lr_scheduler = (lambda opt: torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[91, 136], gamma=0.1))
    bresnet, optim_meter_bresnet = optim.fit(bresnet, data=train_loader, data_val=validation_loader,
                                             lr=1e-1, num_epochs=182, device=device, optimizer='sgd',
                                             weight_decay=1e-4, momentum=0.9, early_stopping=False,
                                             lr_scheduler=lr_scheduler)

    # Print performance
    accuracy = optim.accuracy(bresnet, test_loader=test_loader, device=device)
    print('accuracy:\t', accuracy)

    # Plot training loss
    plt.figure()
    plt.plot(optim_meter_bresnet.train_loss)
