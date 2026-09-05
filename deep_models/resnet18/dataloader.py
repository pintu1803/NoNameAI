from torch.utils.data import DataLoader, Subset
from config import TrainConfig
import torch
from utils import myLog, addLine


def split_dataset_into_loaders(transforms, dataset):

    #Fetch transform methods from imported file
    train_transform = transforms.preprocessing_augmentation_transform()
    test_transform = transforms.preprocessing_transform()

    #define inner methods for transforming train and test samples
    def apply_train_transform(batch):
        batch["image_tensor"] = [train_transform(train_sample) for train_sample in batch["image"]]
        return batch 

    def apply_test_transform(batch):
        batch["image_tensor"] = [test_transform(test_sample) for test_sample in batch["image"]]
        return batch

    #apply transformation on whole sets
    train_transformed = dataset["train"].with_transform(apply_train_transform)
    valid_transformed = dataset["train"].with_transform(apply_test_transform)

    ##############################
    total_size = dataset["train"].num_rows
    #split the entire train set into two parts - train and validation
    train_size = int(total_size * TrainConfig.train_percent)

    #create random indices for entire set size
    random_perm = torch.randperm(total_size, generator=torch.Generator().manual_seed(77))
    train_indices = random_perm[:train_size]
    valid_indices = random_perm[train_size:]

    train_dataset = Subset(train_transformed, train_indices)
    valid_dataset = Subset(valid_transformed, valid_indices)
    test_dataset = dataset["test"].with_transform(apply_test_transform)

    ##############################
    def collate_fn(batch):
        images = torch.stack([item["image_tensor"] for item in batch])  # [B, C, H, W]
        labels = torch.tensor([item["label"] for item in batch])        # [B]
        return images, labels

    train_loader = DataLoader(dataset=train_dataset, shuffle=True,
                              batch_size=TrainConfig.batch_size,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(dataset=valid_dataset, shuffle=True,
                              batch_size=TrainConfig.batch_size,
                              collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True,
                             batch_size=TrainConfig.test_batch_size,
                             collate_fn=collate_fn)

    myLog("Dataloader created for train, validation and testing")
    print(f"Size of loaders = Train:{len(train_loader)}, Valid:{len(valid_loader)}, Test: {len(test_loader)}")

    #Batch validation
    addLine()
    for sample, label in train_loader:
        print("Size of one batch : ", len(sample))
        print("Shape of one batch : ", sample.shape)
        one_sample = sample[0]
        print("One sample shape : ", one_sample.shape)
        print("\nOne sample tensor : ", one_sample)
        print("\nLabel is : ", label[0])
        break
    addLine()
    ###############

    return train_loader, valid_loader, test_loader
