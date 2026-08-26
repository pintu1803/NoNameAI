
import torch
import torchvision.datasets as datasets
from torchvision import transforms


train = datasets.CIFAR10(root="./deep_models/cifar_cnn/dataset", download=True, train=True)

img, label = train[0]

print("Type of img : ",type(img))
print("Size of img : ", img.size)

# transform = transforms.ToTensor()
# train = datasets.CIFAR10(root="./deep_models/cifar_cnn/dataset", download=False, train=True)
