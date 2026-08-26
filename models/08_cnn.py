
import torch
import torchvision.datasets as datasets


d = datasets.CIFAR10(root="./deep_models/cifar_cnn/dataset", download=True, train=True)
