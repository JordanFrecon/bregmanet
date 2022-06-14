# BregmaNet : Bregman Neural Networks

![license](https://img.shields.io/github/license/JordanFrecon/bregmanet)
![release](https://img.shields.io/github/v/release/JordanFrecon/bregmanet?include_prereleases)
![PyPI](https://img.shields.io/pypi/v/bregmanet)

[BregmaNet](https://github.com/JordanFrecon/bregmanet) is a PyTorch library providing multiple [Bregman Neural Networks](https://jordan-frecon.com/download/2022_Frecon_J_p-icml_bnn.pdf).
To date, implemented models cover Bregman variants of multi-layer perceptrons and various residual networks.


**Contributor:** Jordan Frécon (INSA Rouen Normandy, France)

## Table of Contents

1. [Requirements and Installation](#Requirements-and-Installation)
2. [Getting Started](#Getting-Started)
3. [List of Supported Models](#List-of-Supported-Models)
4. [Citation](#Citation)
5. [Contribution and Acknowledgments](#Contribution-and-Acknowledgments)



## Requirements and Installation

### :clipboard: Requirements

- PyTorch version >=1.7.1
- Python version >=3.6
- Torchvision version >=0.8.2


### :hammer: Installation

```
pip install bregmanet
```

In development versions can be found [here](https://test.pypi.org/project/bregmanet/).

## Getting Started

###  :warning: Precautions

* All images should be scaled within the domain range of the activation function.
* MLP models provided work only for 1-dimensional data inputs.
* MLP models are designed without a softmax final layer.
* All models need to be trained first. If you wish to provide your pretrained models, please [contribute](#Contribution-and-Acknowledgments).

### :rocket: Demos

Multiple demo files can be found [there](https://github.com/JordanFrecon/bregmanet) in the *demos* folder. It contains:
- *demo_toy_mlp.py*: training of MLP on the Two-spiral toy dataset.
- *demo_mnist_mlp.py*: training of MLP on the MNIST dataset.
- *demo_cifar10_resnet.py*: training of ResNet20 on the CIFAR-10 dataset.
- *demo_cifar100_resnet.py*: training of ResNet20 on the CIFAR-100 dataset.
- *demo_imagenet_resnet.py*: training of ResNet18 on the ImageNet dataset.



###  :page_with_curl: Loading a Model

To date, all Bregman neural models provided are not trained.
If needed, a training procedure is made available [there](https://github.com/JordanFrecon/bregmanet/) in the *demos/utils* folder.
In order to load a model, proceed as follows.

<details><summary>Multi-Layer Perceptrons</summary><p>

For a *sigmoid*-based MLP with 
- a linear input accepting 1d tensors of size 1024
- 3 hidden layers of size (1024, 1024, 512)
- a linear output layer mapping to 1d tensors of size 10

```python
import bregmanet
model = bregmanet.MLP(activation='sigmoid', num_neurons=[1024, 1024, 512], input_dim=1024, output_dim=10)
```
</p></details>

<details><summary>ResNet</summary><p>

For a BregmanResNet20 with SoftPlus activation function:

```python
import bregmanet
model = bregmanet.bresnet20(activation='softplus')
```

</p></details>


## List of Supported Models

The following list reports all models currently supporting a Bregman variant. 
If you have any issue with one of them or wish to provide your own, please [contact us](mailto:jordan.frecon@gmail.com).

- MLP
- ResNet18
- ResNet20
- ResNet32
- ResNet34
- ResNet44
- ResNet56
- ResNet101
- Resnet110
- ResNet152
- Resnet1202
- ResNeXt50_32x4d
- ResNeXt101_32x8d
- WideResNet50_2
- WideResnet101_2


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

Jordan Frecon would like to express his gratitude to the [Department of Computational Statistics and Machine Learning](https://www.iit.it/web/computational-statistics-and-machine-learning) (IIT, Genova, Italy) where part of this work was conducted during his postdoctoral position. The authors gratefully acknowledge the financial support of the French Agence Nationale de la Recherche (ANR), under grant ANR-20-CHIA-0021-01 ([project RAIMO](https://chaire-raimo.github.io)).

The proposed BregmanResNets for CIFAR-10 are based on a rework of the ResNet implementation of [Yerlan Idelbayev](https://github.com/akamaster/pytorch_resnet_cifar10).
Other ResNet models are devised by hinging upon the official PyTorch/TorchVision repository. For more information, please refer to:
- ResNet: ["Deep Residual Learning for Image Recognition"](https://arxiv.org/pdf/1512.03385.pdf) 
- ResNeXt: ["Aggregated Residual Transformation for Deep Neural Networks"](https://arxiv.org/pdf/1611.05431.pdf)
- WideResNet: ["Wide Residual Networks"](https://arxiv.org/pdf/1605.07146.pdf)

All kind of contributions are welcome, do not hesitate to [contact us!](mailto:jordan.frecon@gmail.com)
