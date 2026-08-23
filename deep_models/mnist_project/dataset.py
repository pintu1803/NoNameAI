
"""
Author: Pintu Saini
Date: 23-08-2026

MNIST image classification Deep Learning model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# torch vision is the library for datasets
from torchvision import transforms
from torch.utils.data import DataLoader
#random split for train and validation
from torch.utils.data import random_split
#dataset download or not
from config import DatasetConfig

def get_dataloaders():
    dataset = torchvision.datasets.MNIST

    #download data first, then make download False and use offline dataset
    dataset = torchvision.datasets.MNIST(root="./deep_models/mnist_project/train_data", 
                                         train=True, download=DatasetConfig.downloadTrain)

    test_dataset = torchvision.datasets.MNIST(root="./deep_models/mnist_project/test_data", 
                                              train=False, download=DatasetConfig.downloadTest)


    #analyse the dataset briefly
    print("\nBefore transform")

    #raw data type
    #PIL - python image library
    #Image is PIL type -> do not have shape, min, max attributes
    image, label = dataset[0]
    print("Length of dataset : ", len(dataset))
    print("Image item : ", image, label)
    print("Types : ", type(image), type(label))


    print("\nNow transform raw dataset into tensors")

    #create transformer
    #accepts a list of parameters
    transform = torchvision.transforms.Compose([transforms.ToTensor(),])

    #reload dataset and convert into tensors
    #transform converts PIL object -> tensor
    #normalized values between 0 t0 1
    dataset = torchvision.datasets.MNIST(root="./deep_models/mnist_project/train_data", 
                                        train=True, download=False,
                                        transform=transform)

    print("\nAfter dataset transformation")
    print("Type : ", type(dataset[0][0]))

    #image is grayscale, so one channel exists
    image, label = dataset[0]
    print("Image shape : ", image.shape)

    flat_image = torch.flatten(image)
    print("Min of image : ", min(flat_image))
    print("Max of image : ", flat_image.max())
    print("Mean of image tensor : ", torch.mean(flat_image).item())
    print("Std of image tensor : ", torch.std(flat_image).item())

    print("\nPerform normalisation of image")

    #Transform such that image values get centralized
    #Use mean 0.5 and std 0.5 to normalize the datapoints
    #output = (x-mean)/std
    n_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.5,),
                                        (0.5,)
                                    )])

    #Reload dataset with new transform
    dataset = torchvision.datasets.MNIST(root="./deep_models/mnist_project/train_data", 
                                        train=True, download=False,
                                        transform=n_transform)

    test_dataset = torchvision.datasets.MNIST(root="./deep_models/mnist_project/test_data", 
                                        train=False, download=False,
                                        transform=n_transform)

    image, label = dataset[0]
    flat_image = torch.flatten(image)

    print("\nAfter tensor normalisation\n")

    print("Image shape : ", image.shape)
    print("Min of image : ", flat_image.min())
    print("Max of image : ", flat_image.max())
    print("Mean of image tensor : ", torch.mean(flat_image).item())
    print("Std of image tensor : ", torch.std(flat_image).item())

    ##################################################################

    #Purpose of seed
    """
    so everytime my python file runs, it always creates the same split for training and validation datasets. 
    and then loader can shuffle within that subset for every epoch. 
    it ensures my training set is not changed for future training runs also.
    """
    #Split for train and validation
    N = len(dataset)
    train_size = int(N*0.9)
    val_size = int(N*0.1)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                generator=torch.Generator().manual_seed(77))

    print("\nTraining dataset size : ", train_size)
    print("Validatoin dataset size : ", val_size)
    print("Testing dataset size : ", len(test_dataset))

    ##################################################################

    #Dataloader
    batch_size = 64
    print("\nData loader will shuffle entire set and make batches for every epoch")

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Train: batch_size {batch_size}, batch count {len(train_loader)}")

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    print(f"Validate: batch_size {batch_size}, batch count {len(val_loader)}")

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    print(f"Test: batch_size {batch_size}, batch count {len(test_loader)}")

    return train_loader, val_loader, test_loader




