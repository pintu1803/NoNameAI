
import torch
import torchvision.datasets as datasets
from torchvision import transforms

"""
Images are tensors
RGB = 3 channels
Tensor shape = (N, C, H, W)
N = batch size, C = channel, H = height, W = width
Flattening destroys spatial relationships
Convolution uses a small kernel to detect patterns
Edge detection example

---------------
Image
↓
One Filter
↓
One Output Image

That output image is called a Feature Map.
---------------
"""

train = datasets.CIFAR10(root="./deep_models/cifar_cnn/dataset", download=True, train=True)

img, label = train[0]

print("Type of img : ",type(img))
print("Size of img : ", img.size)

# transform = transforms.ToTensor()
# train = datasets.CIFAR10(root="./deep_models/cifar_cnn/dataset", download=False, train=True)


