
import config
import utils
import resnet18
import dataset
import transform
import dataloader
import train


def main():
    model = ""
    # model = resnet18.load_resnet18()
    data_set = dataset.load_data_from_cache()
    train_loader, valid_loader, test_loader = dataloader.split_dataset_into_loaders(transform, data_set)
    train.train_model(train_loader, valid_loader, model)

if __name__ == "__main__":
    main()