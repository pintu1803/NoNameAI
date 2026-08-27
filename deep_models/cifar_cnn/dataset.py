
import torch
import torchvision.datasets as datasets
from torchvision import transforms
from config import BASE_DIR
from config import TrainConfig
from torch.utils.data import random_split
from torch.utils.data import DataLoader

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

Each output learns one feature.
N filters learn N features about the image.
"""

def prepare_dataset():
    dataset_path = BASE_DIR / "train_data"
    print("\nDataset download to or upload from : ", dataset_path)

    ## transform1 just converts values in tensors in range [0, 1]
    ## transform2 normalizes values in range [-1, 1] using mean = 0.5, std = 0.5

    # transform1 = transforms.ToTensor()
    transform2 = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])
    train = datasets.CIFAR10(root=dataset_path, 
                            download=False, train=True,
                            transform=transform2)

    img, label = train[0]
    total_size = len(train)

    print("Type of img : ",type(img)) # <class 'torch.Tensor'>
    print("Size of img : ", img.shape) # torch.Size([3, 32, 32])
    print("Size of dataset : ", len(train)) # 50000

    flat_img = torch.flatten(img)
    print("Number of pixel in image : ", flat_img.shape) # torch.Size([3072])
    print("Min value of img pixel : ", torch.min(flat_img)) # tensor(-1.)
    print("Max value of img pixel : ", torch.max(flat_img)) # tensor(1.)
    # print("Mean of image : ", torch.mean(flat_img)) # tensor(-0.1886)
    # print("Std of image : ", torch.std(flat_img)) # tensor(0.4077)

    ###########################################################
    train_size = int(total_size * TrainConfig.train_percent)
    valid_size = int(total_size * TrainConfig.valid_percent)
    train_set, valid_set = random_split(train, [train_size, valid_size],
                                        generator=torch.Generator().manual_seed(77)) 

    print("\nSize of training set : ", len(train_set))
    print("Size of validation set : ", len(valid_set))

    ###########################################################
    train_loader = DataLoader(train_set, batch_size=TrainConfig.batch_size, 
                            shuffle=TrainConfig.shuffle_train_set, 
                            drop_last=TrainConfig.drop_last_in_train_set)
    print("\nNumber of batches in training set : ", len(train_loader))

    valid_loader = DataLoader(valid_set, shuffle=TrainConfig.shuffle_train_set,
                              batch_size=TrainConfig.batch_size,
                              drop_last=TrainConfig.drop_last_in_train_set)

    return train_loader, valid_loader