
import torch
import torchvision.datasets as datasets
from torchvision import transforms
from config import PATH
from config import TrainConfig
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Subset

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

def prepare_testing_dataset():
    dataset_path = PATH.TEST_DATA_DIR
    print("\nTest dataset download to or upload from : ", dataset_path)

    transform2 = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])
    test = datasets.CIFAR10(root=dataset_path, 
                            download=False, train=False,
                            transform=transform2)

    test_loader = DataLoader(test, batch_size=TrainConfig.batch_size)
    print("\nNumber of batches in testing set : ", len(test_loader))

    return test_loader


def prepare_training_dataset():
    dataset_path = PATH.TRAIN_DATA_DIR
    print("\nDataset download to or upload from : ", dataset_path)

    ###########################################################
    full_dataset = datasets.CIFAR10(root=dataset_path, 
                                    download=False, train=True,
                                    transform=None)

    total_size = len(full_dataset)
    print("Size of dataset : ", total_size) # 50000

    random_perm = torch.randperm(total_size, generator=torch.Generator().manual_seed(77))
    train_size = int(total_size * TrainConfig.train_percent)

    ###########################################################
    ## transform1 just converts values in tensors in range [0, 1]
    ## transform2 normalizes values in range [-1, 1] using mean = 0.5, std = 0.5

    # transform1 = transforms.ToTensor()
    validation_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(TrainConfig.normMean, TrainConfig.normVar)])
    train_transform_data_augment = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomRotation(10),
                                                        transforms.ColorJitter(contrast=0.2,
                                                                               saturation=0.2,
                                                                               brightness=0.2,
                                                                               hue=0.02),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(TrainConfig.normMean, TrainConfig.normVar)])

    train_dataset = datasets.CIFAR10(root=dataset_path, 
                                    download=False, train=True,
                                    transform=train_transform_data_augment)
    valid_dataset = datasets.CIFAR10(root=dataset_path, 
                                    download=False, train=True,
                                    transform=validation_transform)

    train_indices = random_perm[:train_size]
    valid_indices = random_perm[train_size:]

    train_set = Subset(train_dataset, train_indices)
    valid_set = Subset(valid_dataset, valid_indices)

    img, label = train_set[0]

    print("Type of img : ",type(img)) # <class 'torch.Tensor'>
    print("Size of img : ", img.shape) # torch.Size([3, 32, 32])

    print("\nType of label : ", type(label))
    print("Value of label : ", label)

    flat_img = torch.flatten(img)
    print("\nNumber of pixel in image : ", flat_img.shape) # torch.Size([3072])
    print("Min value of img pixel : ", torch.min(flat_img)) # tensor(-1.)
    print("Max value of img pixel : ", torch.max(flat_img)) # tensor(1.)
    # print("Mean of image : ", torch.mean(flat_img)) # tensor(-0.1886)
    # print("Std of image : ", torch.std(flat_img)) # tensor(0.4077)

    ###########################################################
    train_loader = DataLoader(train_set, batch_size=TrainConfig.batch_size, 
                            shuffle=TrainConfig.shuffle_train_set, 
                            drop_last=TrainConfig.drop_last_in_train_set)
    print("\nNumber of batches in training set : ", len(train_loader))

    print("Type of train_loader : ", type(train_loader))

    for image_batch, label_batch in train_loader:
        print("\nType of image batch : ", type(image_batch))
        print("Shape of image batch : ", image_batch.shape)
        print("Type of batch label : ", type(label_batch))
        print("Shape of batch label : ", label_batch.shape)
        break

    valid_loader = DataLoader(valid_set, shuffle=TrainConfig.shuffle_train_set,
                              batch_size=TrainConfig.batch_size,
                              drop_last=TrainConfig.drop_last_in_train_set)

    return train_loader, valid_loader



###########
###########

def main():
    train_loader, valid_loader = prepare_training_dataset()
    test_loader = prepare_testing_dataset()
    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    main()

###########
###########
