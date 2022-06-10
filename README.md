# BregmaNet : Bregman Neural Networks

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[BregmaNet](https://github.com/JordanFrecon/bregmanet) is a PyTorch library providing multiple Bregman neural networks. To date, implemented models cover Bregman variants of multi-layer perceptrons and various residual networks.


## Table of Contents

1. [Requirements and Installation](#Requirements-and-Installation)
2. [Getting Started](#Getting-Started)
3. [Citation](#Citation)
4. [Contribution](#Contribution-and-Acknowledgments)



## Requirements and Installation

### :clipboard: Requirements

- PyTorch version >=1.7.1
- Python version >=3.6
- Torchvision version >=0.8.2


### :hammer: Installation

```
pip install bregmanet
```

Development versions can be found [here](https://test.pypi.org/project/bregmanet/).

## Getting Started

###  :warning: Precautions

* All images should be scaled to the domain range of the activation function.
* MLP models provided are designed for 1-dimensional data inputs.


###  :page_with_curl: Models

In order to load untrained Bregman neural models, proceed as follows.

**Bregman Multi-Layer Perceptrons**

```python
import bregmanet
model = bregmanet.MLP(activation_name='sigmoid', num_neurons=[1024, 1024, 512], input_dim=1024, output_dim=10)
```


**Bregman Residual Networks**

For instance, a BregmanResNet20 with SoftPlus activation function can be defined as :

```python
import bregmanet
model = bregmanet.bresnet20(activation='softplus')
```


### :rocket: Demos

Multiple demo files can be found [there](https://github.com/JordanFrecon/bregmanet) in the *demo* folder. It contains:
- *demo_toy_mlp.py*: training of MLPs on the Two-spiral toy dataset.
- *demo_cifar10_resnet.py*: training of ResNet20 on the CIFAR-10 dataset.
- *demo_cifar100_resnet.py*: training of ResNet20 on the CIFAR-100 dataset.
- *demo_imagenet_resnet.py*: training of ResNet18 on the ImageNet dataset.



## Citation

If you use this package, please cite the following work:

```
@inproceedings{2022_Frecon_J_p-icml_bregmanet,
  title = {{Bregman Neural Networks}},
  author = {Frecon, Jordan and Gasso, Gilles and Pontil, Massimiliano and Salzo, Saverio},
  url = {https://hal.archives-ouvertes.fr/hal-03132512},
  series    = {Proceedings of Machine Learning Research},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning,
               {ICML} 2022, 17-23 July 2022, Baltimore, USA},
  year = {2022},
}

```


## Contribution and Acknowledgments

The proposed BregmanResNet for CIFAR-10 is based on a rework of the ResNet implementation of [Yerlan Idelbayev](https://github.com/akamaster/pytorch_resnet_cifar10).

All kind of contributions are welcome, do not hesitate to contact us!
