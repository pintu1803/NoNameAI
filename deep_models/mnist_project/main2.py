#Test the saved model.

from dataset import get_dataloaders
from config import ModelConfig
from model import MNISTModel
from train import train_model
from test import test_model

from config import BASE_DIR
import torch


def main():
    _, _, test_loader = get_dataloaders()

    config = ModelConfig()

    modelObject = MNISTModel(config)

    #Training is not required, use saved weights
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    MODEL_PATH = CHECKPOINT_DIR / "mnist_model.pth"
    model = modelObject.deepModel
    model.load_state_dict(torch.load(MODEL_PATH))

    test_model(modelObject, test_loader)

if __name__ == "__main__":
    main()