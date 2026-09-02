#CIFAR 10 

import dataset
from model import cnn_model1, cnn_model2
import train
import test
import config
from config import PATH
import torch

def main():
    train_loader, val_loader, test_loader = dataset.main()

    # model-1 gives overall 62% accuracy
    # modelObj = cnn_model1(config.ModelConfig)
    # model-2 has better architecture, use it now.
    modelObj = cnn_model2(config.ModelConfig)

    model = modelObj.model
    model.load_state_dict(torch.load(PATH.SAVED_MODEL_FULL_PATH))

    train.train_cifar_model(modelObj, train_loader, val_loader)

    test.test_model(modelObj, test_loader)


############################
if __name__ == "__main__":
    main()