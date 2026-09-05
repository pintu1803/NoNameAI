from torch.utils.data import DataLoader, Subset
from config import TrainConfig
import torch
from torch import permute
from utils import myLog


def split_dataset_into_loaders(transforms, dataset):

    #Fetch transform methods from imported file
    train_transform = transforms.preprocessing_augmentation_transform()
    test_transform = transforms.preprocessing_transform()

    #define inner methods for transforming train and test samples
    def apply_train_transform(train_sample):
        train_sample["image_tensor"] = train_transform(train_sample["image"])
        return train_sample 

    def apply_test_transform(test_sample):
        test_sample["image_tensor"] = test_transform(test_sample["image"])
        return test_sample

    #apply transformation on whole sets
    train_transformed = dataset["train"].with_transform(apply_train_transform)
    valid_transformed = dataset["train"].with_transform(apply_test_transform)

    ##############################
    total_size = dataset["train"].num_rows
    #split the entire train set into two parts - train and validation
    train_size = int(total_size * TrainConfig.train_percent)
    valid_size = int(total_size * TrainConfig.valid_percent)

    #create random indices for entire set size
    random_perm = torch.randperm(total_size, generator=torch.Generator().manual_seed(77))
    train_indices = random_perm[:train_size]
    valid_indices = random_perm[:valid_size]

    train_dataset = Subset(train_transformed, train_indices)
    valid_dataset = Subset(valid_transformed, valid_indices)
    test_dataset = dataset["test"].with_transform(apply_test_transform)

    ##############################
    train_loader = DataLoader(dataset=train_dataset, shuffle=True,
                              batch_size=TrainConfig.batch_size)
    valid_loader = DataLoader(dataset=valid_dataset, shuffle=True,
                              batch_size=TrainConfig.batch_size)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True,
                             batch_size=TrainConfig.test_batch_size)

    myLog("Dataloader created for train, validation and testing")
    print(f"Size of loaders = Train:{len(train_loader)}, Valid:{len(valid_loader)}, Test: {len(test_loader)}")

    return train_loader, valid_loader, test_loader
