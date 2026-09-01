from config import TrainConfig
from config import PATH
import torch.nn as nn
import torch.optim as optim
import utils
from datetime import datetime

def train_cifar_model(modelObj, train_loader, valid_loader):
    epochs = TrainConfig.epoch_count
    checkpoint = TrainConfig.epoch_checkpoint

    #fetch model from object
    model = modelObj.model

    #define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=TrainConfig.learning_rate,
                            weight_decay=TrainConfig.weigth_decay)

    #make list of loss and accuracy
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    ## Inner method for training
    def train():
        model.train()

        epoch_train_loss = 0
        epoch_train_correct_pred = 0
        epoch_train_set_size = 0
        batch_count = 0

        for idx, (image, label) in enumerate(train_loader):
            prediction = model(image)
            loss = loss_fn(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.mean().item()
            epoch_train_correct_pred += (label == prediction.argmax(dim=-1)).sum()
            epoch_train_set_size += len(label)
            batch_count += 1

        #eval epoch loss and accuracy and store
        epoch_avg_train_loss = epoch_train_loss/batch_count
        epoch_train_accuracy = epoch_train_correct_pred*100/epoch_train_set_size

        train_losses.append(epoch_avg_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        return epoch_avg_train_loss, epoch_train_accuracy

    ## Inner method for validating
    def validate():
        model.eval()

        epoch_valid_loss = 0
        epoch_valid_correct_pred = 0
        epoch_valid_set_size = 0
        batch_count = 0
        
        for idx, (image, label) in enumerate(valid_loader):
            pred = model(image)
            loss = loss_fn(pred, label)

            epoch_valid_loss += loss.mean().item()
            epoch_valid_correct_pred += (label == pred.argmax(dim=-1)).sum().item()
            epoch_valid_set_size += len(label)
            batch_count += 1

        #eval epoch loss and accuracy and store
        epoch_avg_val_loss = epoch_valid_loss/batch_count
        epoch_val_accuracy = epoch_valid_correct_pred*100/epoch_valid_set_size

        val_losses.append(epoch_avg_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        return epoch_avg_val_loss, epoch_val_accuracy

    ## Full Training loop begins 
    print("\nTRAINING BEGINS\n")

    #Training and validation epochs start
    best_val_acc = 0
    CHECKPOINT_DIR = PATH.CHECKPOINT_DIR
    MODEL_NAME = TrainConfig.SAVE_MODEL_NAME

    for epoch in range(epochs):

        epoch_avg_train_loss, epoch_train_accuracy = train()
        epoch_avg_val_loss, epoch_val_accuracy = validate()

        if epoch_val_accuracy > best_val_acc:
            best_val_acc = epoch_val_accuracy
            ### Save the model
            utils.save_trained_parameters(model, CHECKPOINT_DIR, MODEL_NAME)
            print(f"\nSaving the BEST model so far, at epoch : {epoch + 1}, acc : {best_val_acc : .3f}")

        #Print epoch loss and acc for train and valid
        print(f"\nEpoch : {epoch+1}")
        print(f"Training loss : {epoch_avg_train_loss : .3f}, acc : {epoch_train_accuracy : .3f}")
        print(f"Validation loss : {epoch_avg_val_loss : .3f}, acc : {epoch_val_accuracy : .3f}")

        
    # Dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    PLOT_DIR = PATH.PLOT_DIR
    LOSS_PLOT_NAME = PLOT_DIR / "train_vs_val_loss_curve_{timestamp}.png"
    ACC_PLOT_NAME = PLOT_DIR / "train_vs_val_acc_curve_{timestamp}.png"

    #### Training completed ####
    utils.plot_train_vs_validation_curve(PLOTS_DIR=PLOT_DIR, 
                                         plot_name=LOSS_PLOT_NAME, 
                                         first_item=train_losses,
                                         first_label="Training Loss", 
                                         second_item=val_losses,
                                         second_label="Validation Loss",
                                         xlabel="Epochs", ylabel="Loss")

    utils.plot_train_vs_validation_curve(PLOTS_DIR= PLOT_DIR,
                                         plot_name=ACC_PLOT_NAME,
                                         first_item=train_accuracies,
                                         first_label="Training Accuracy",
                                         second_item=val_accuracies,
                                         second_label="Validation Accuracy",
                                         xlabel="Epochs",
                                         ylabel="Accuracy")
