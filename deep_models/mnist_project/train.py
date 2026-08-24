##Train configs are stored in config.py

from config import TrainConfig
import torch
import os

def train_model(modelObj, train_loader, val_loader):
    epochs = TrainConfig.epoch_count
    checkpoint = TrainConfig.epoch_checkpoint

    modelObj.showModel()

    loss_fn = modelObj.loss_fn()

    optimizer = modelObj.optimizer()

    model = modelObj.deepModel

    #Training loss and Validation loss
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    #TRAIN and Validate
    for epoch in range(epochs):

        #Training mode ON
        model.train()

        train_loss = 0
        for image, label in train_loader:

            pred = modelObj.forward(image)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #add loss for all batches
            train_loss += loss

        #Validation mode ON
        model.eval()

        val_loss = 0
        for image, label in val_loader:
            pred = modelObj.forward(image)
            loss = loss_fn(pred, label)  
            val_loss += loss   

        #find average loss of all batches
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)

        #store the losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        #print for every checkpoint-th epoch
        if epoch % checkpoint == 0:
            print(f"\nLoss computed - epoch : {epoch}")
            print(f"Avg Train Loss : {avg_train_loss : .3f}, Avg Validation Loss : {avg_val_loss : .3f}")

    ### Training finished.
    ### save the model
    save_model(model)


def save_model(model):
    print("\nMake checkpoints dir")

    os.makedirs("checkpoints", exist_ok=True)

    torch.save(model.state_dict(),
               "./deep_models/mnist_project/checkpoints/mnist_model.pth"
               )

    print("\nModel state_dict : ", model.state_dict())