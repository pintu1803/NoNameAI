
import config
import utils
import resnet18
import dataset
import transform
import dataloader


def main():
    model = resnet18.load_resnet18()
    data_set = dataset.load_data_from_cache()
    train_loader, valid_loader, test_loader = dataloader.split_dataset_into_loaders(transform, data_set)
    

if __name__ == "__main__":
    main()