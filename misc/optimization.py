from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import os


class EarlyStopping:
    """
    Early stops the training if validation loss/accuracy doesn't improve after a given patience.
    See https://github.com/Bjarten/early-stopping-pytorch]
    """

    def __init__(self, patience=7, verbose=False, delta=1e-5, path='checkpoint', trace_func=print, criterion='loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
            criterion (string): criterion used for early stopping ('loss' or 'accuracy')
                            Default: loss
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.criterion = criterion.lower()
        self.val_best = np.Inf if self.criterion == 'loss' else 0
        self.delta = delta
        self.path = path + str(np.random.randint(1000)) + '.pt'
        self.trace_func = trace_func

    def __call__(self, val_loss, val_acc, model):

        score = -val_loss if self.criterion == 'loss' else val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, val, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            if self.criterion == 'loss':
                val = -val
                self.trace_func(
                    f'Validation loss decreased ({self.val_best:.6f} --> {val:.6f}).  Saving model ...')
            else:
                self.trace_func(
                    f'Validation accuracy increased ({self.val_best:.6f} --> {val:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_best = val

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))
        return model


class LineSearch:
    """
    Reduce the step-size in order to yield a decreasing training loss / increasy training accuracy
    """

    def __init__(self, opt, delta=0, path='ls_checkpoint.pt', trace_func=print, criterion='loss', verbose=False):
        """
        Args:
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
            criterion (string): criterion used for early stopping ('loss' or 'accuracy')
                            Default: loss
        """
        self.best_score = None
        self.early_stop = False
        self.criterion = criterion.lower()
        self.val_best = np.Inf if self.criterion == 'loss' else 0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.verbose = verbose
        #self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, verbose=True)
        lmbda = lambda epoch: 0.95
        self.lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=opt, lr_lambda=lmbda, verbose=True)

    def __call__(self, loss, acc, model):

        score = loss if self.criterion == 'loss' else -acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score > self.best_score + self.delta:
            if self.verbose:
                if self.criterion == 'loss':
                    self.trace_func(
                        f'Training loss increased ({self.best_score:.6f} --> {score:.6f}).  Load previous model ...')
                else:
                    score = -score
                    self.trace_func(
                        f'Training accuracy decreased ({self.best_score:.6f} --> {score:.6f}).  Load previous model ...')
            model = self.load_checkpoint(model)
            self.lr_scheduler.step()#(-1)
        else:
            self.best_score = score
            self.save_checkpoint(model)

        return model

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))
        return model


class OptimizationMeter(object):
    def __init__(self, *args):
        for val in args:
            setattr(self, val, [])

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                tmp = getattr(self, key)
                tmp.append(value)
                setattr(self, key, tmp)
            else:
                setattr(self, key, [value])


def accuracy_criterion(prediction, true_label, label_reduction=None):
    if label_reduction is None:
        target = prediction.argmax(dim=1)
    else:
        target = torch.pow(prediction - label_reduction, 2).argmin(dim=1)
    return float(target.eq(true_label).sum())


def fit(model, data, data_val=None, loss_func=None, lr=.1, num_epochs=182, optimizer='sgd', device=torch.device('cpu'),
        weight_decay=0, momentum=0, early_stopping=True, label_reduction=None, lr_scheduler=None, lambda_reg=None,
        delta=0, patience=10, verbose=True):
    """Learn the neural network"""

    # Loading dataset on device
    model = model.to(device)

    # Trick to handle regression case
    label_redux = None if label_reduction is None else torch.Tensor(label_reduction).to(device=device)

    # Criterion
    if loss_func is None:
        loss_func = nn.CrossEntropyLoss()
    loss_func = loss_func.to(device=device)

    # Optim meters
    optim_meter = OptimizationMeter()

    # Optimizer
    if optimizer.lower() == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer.lower() == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    # Lr Scheduler
    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(opt)

    # Early stopping
    check_early_stopping = True if (data_val is not None and early_stopping) else False
    if check_early_stopping:
        early_stopping = EarlyStopping(patience=patience, verbose=verbose, criterion='accuracy', delta=delta)

    # Regularizer
    if lambda_reg is not None:
        prox_l1 = torch.nn.Softshrink(lambd=lr*lambda_reg)

    # Iterate through dataset
    if verbose:
        bar = trange(int(num_epochs))
    else:
        bar = range(int(num_epochs))
    for epoch in bar:

        ###################
        # train the model #
        ###################
        train_losses = 0
        train_acc = 0
        model.train()
        for x, y in data:
            # Assign to the device
            x, y = x.to(device=device), y.to(device=device)

            # Update model
            for param in model.parameters():
                param.grad = None
            prediction = model(x)
            loss = loss_func(prediction, y)
            loss.backward()
            opt.step()

            # Keep track of optimization meters
            with torch.no_grad():

                # Soft thresholding
                if lambda_reg is not None:
                    for name, var in model.named_parameters():
                        if 'lin' in name:
                            var.copy_(prox_l1(var))
                            train_losses += float(lambda_reg*torch.sum(torch.abs(var)))

                # Optimization values
                train_acc += accuracy_criterion(prediction, y, label_reduction=label_redux)
                train_losses += float(loss)

                if np.isnan(train_losses):
                    break
                del prediction, loss
        train_acc = train_acc / data.dataset.__len__()
        optim_meter.update(train_loss=train_losses, train_accuracy=train_acc)

        ######################
        # validate the model #
        ######################
        if check_early_stopping:
            with torch.no_grad():
                model.eval()
                valid_losses = 0
                valid_acc = 0
                for x, y in data_val:
                    # Assign to the device
                    x, y = x.to(device=device), y.to(device=device)
                    prediction = model(x)

                    # Update model
                    loss = loss_func(model(x), y)
                    valid_losses += float(loss)
                    valid_acc += accuracy_criterion(prediction, y, label_reduction=label_redux)
                    del prediction, loss
                valid_acc = valid_acc / data_val.dataset.__len__()
                optim_meter.update(validation_loss=valid_losses, validation_accuracy=valid_acc)

                # Early stopping
                early_stopping(valid_losses, valid_acc, model)
                if early_stopping.early_stop and verbose:
                    print("Early stopping")
                    break

        if verbose:
            bar.set_description('EPOCH: %d - loss: %.3f | acc: %.3f' % (epoch + 1, train_losses, train_acc))

        if lr_scheduler is not None:
            lr_scheduler.step()
            if lambda_reg is not None:
                prox_l1 = torch.nn.Softshrink(lambd=lr_scheduler.get_lr()*lambda_reg)

    # Remove potential checkpoints and load best model
    if check_early_stopping:
        model = early_stopping.load_checkpoint(model)
        #os.remove('checkpoint.pt')
        os.remove(early_stopping.path)

    # Output
    return model, optim_meter


def accuracy(model, test_loader, device=torch.device('cpu'), labels=None, num_samples=None):
    """Compute the accuracy of the model"""
    model = model.to(device=device).eval()
    label_redux = None if labels is None else torch.Tensor(labels).to(device=device)
    correct = 0
    if num_samples is None:
        num_samples = test_loader.dataset.__len__()
    for x, y in test_loader:
        x, y = x.to(device=device), y.to(device=device)

        correct += accuracy_criterion(model(x), y, label_reduction=label_redux)
        #if labels is None:
        #    target = model(x).argmax(dim=1)
        #else:
        #    target = torch.pow(model(x) - tensor_labels, 2).argmin(dim=1)
        #correct += target.eq(y).sum()

    return correct / num_samples


def top_accuracy(model, test_loader, device=torch.device('cpu'), k=2, softmax=True):
    """Compute the k-top accuracy of the model"""
    model = model.to(device=device).eval()
    num_samples = test_loader.dataset.__len__()
    correct = 0
    dist = 0
    num_correct = 0
    do_softmax = torch.nn.Softmax()
    for x, y in test_loader:
        x, y = x.to(device=device), y.to(device=device)

        if softmax:
            pred = do_softmax(model(x))
        else:
            pred = model(x)
        values, indices = torch.topk(pred, k=k, dim=1)
        for (yt, it, vt) in zip(y, indices, values):
            correct += float((yt in it))
            if yt == it[0]:
                num_correct += 1
                dist += float(torch.abs(vt[0]-vt[1]))

    return correct / num_samples, dist / num_correct
