from dataset import get_dataloaders
from config import ModelConfig
from model import MNISTModel
from train import train_model
from test import test_model


def main():
    train_loader, val_loader, test_loader = get_dataloaders()

    config = ModelConfig()

    modelObject = MNISTModel(config)

    train_model(modelObject, train_loader, val_loader)

    test_model(modelObject, test_loader)

if __name__ == "__main__":
    main()