#CIFAR 10 

import dataset
import model
import train
import config

def main():
    train_loader, val_loader, test_loader = dataset.main()

    modelObj = model.cnn_model(config.ModelConfig)

    train.train_cifar_model(modelObj, train_loader, val_loader)






if __name__ == "__main__":
    main()