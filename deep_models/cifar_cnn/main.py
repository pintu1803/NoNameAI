#CIFAR 10 

import dataset
from model import cnn_model1, cnn_model2
import train
import test
import config

def main():
    train_loader, val_loader, test_loader = dataset.main()

    # model-1 gives overall 62% accuracy
    # modelObj = cnn_model1(config.ModelConfig)
    modelObj = cnn_model2(config.ModelConfig)

    train.train_cifar_model(modelObj, train_loader, val_loader)

    test.test_model(modelObj, test_loader)


############################
if __name__ == "__main__":
    main()