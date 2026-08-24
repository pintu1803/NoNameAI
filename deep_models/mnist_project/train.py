##Train configs are stored in config.py

from config import TrainConfig
import torch
import os
import matplotlib.pyplot as plt


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
    train_accs = []
    val_accs = []

    #TRAIN and Validate
    for epoch in range(epochs):

        #Training mode ON
        model.train()

        #accuracy and loss variables
        train_correct = 0
        train_total = 0
        train_loss = 0

        #iterate all batches in one epoch
        for image, label in train_loader:

            pred = modelObj.forward(image)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #add loss for all batches
            #batch loss is avg loss, epoch loss we calculate as avg
            train_loss += loss

            _, predicted = torch.max(pred, 1)
            train_total += label.size(0)#tuple index 0
            train_correct += (predicted == label).sum().item()
            

        #Validation mode ON
        model.eval()

        val_correct = 0
        val_total = 0
        val_loss = 0
        for image, label in val_loader:
            pred = modelObj.forward(image)
            loss = loss_fn(pred, label)  
            val_loss += loss

            _, predicted = torch.max(pred, 1)
            val_total += label.size(0)
            val_correct += (predicted == label).sum().item()

        #find average loss of all batches
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)

        #store the losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        #compute the accuracy
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        #store the accuracy
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        #print for every checkpoint-th epoch
        if epoch % checkpoint == 0:
            print(f"\nLoss computed - epoch : {epoch}")
            print(f"Avg Train Loss : {avg_train_loss : .3f}, Avg Validation Loss : {avg_val_loss : .3f}")

    ### Training finished.
    ### save the model
    save_model(model)

    ### Plot curves
    plot_loss_curve(train_losses, val_losses)
    plot_accuracy_curve(train_accs, val_accs)


#Save model
def save_model(model):
    print("\nMake checkpoints dir")

    os.makedirs("checkpoints", exist_ok=True)

    torch.save(model.state_dict(),
               "./deep_models/mnist_project/checkpoints/mnist_model.pth"
               )

    print("\nModel state_dict : ", model.state_dict())


#Loss plot
def plot_loss_curve(train_losses, val_losses):
    #save in plots dir
    os.makedirs("plots", exist_ok=True)
    
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()

    plt.savefig("./deep_models/mnist_project/plots/loss_curve.png",
                dpi=300,
                bbox_inches="tight")

    plt.show(block=False)
    plt.pause(10)
    plt.close()

#Accuracy plot
def plot_accuracy_curve(train_acc, val_acc):
    #save in plots dir
    os.makedirs("plots", exist_ok=True)
    
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()

    plt.savefig("./deep_models/mnist_project/plots/accuracy_curve.png",
                dpi=300,
                bbox_inches="tight")

    plt.show(block=False)
    plt.pause(10)
    plt.close()