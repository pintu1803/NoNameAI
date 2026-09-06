import torch.nn as nn
import torch
from config import TrainConfig, PATH
from utils import myLog, addLine, newLine
import utils
from datetime import datetime
import torch.optim as optim
import os

"""
Note to self:
In actual dataset, labels are plain int values, 
and we don't preprocess and convert labels into tensors.
So they stay int during batching, and final batch is just 1-D tensor of int lablels. 
Dataloader makes the batch of these int labels and hands over the list to collate_fn
which converts them into tensor. list -> 1D Tensor

Prediction is a tensor of size = number of classes. 
"""

#Define loss function outside, in order to import in test
loss_fn = nn.CrossEntropyLoss()

def train_model(train_loader, valid_loader, model):
    """
    In this module we train our resnet18 model on our food dataset.
    Loss function is chosen as cross entropy loss as it's best for multi-class classification.
    Optimizer is chosen as adamW which provides weight decay facility
    Cosine Annealing LR scheduler is chosen to smoothly reduce the lr over epochs.

    Training loss is averaged for all the batches over an epoch.
    Training accuracy is also evaluated for one whole epoch.

    Validation loss is averaged for all the batches over an epoch.
    Validation accuracy is also evaluated for one entire epoch.

    Since we are using pretrained resnet weights, update them carefully and minutely.
    With Backbone frozen and only FC unfreezed and trainable -
    we use 1e-3 lr and decay values.
    For everything unfreezed and trainable - we use 1e-4 scale lr values.

    This module performs below operations in given order:
    1. Define default fresh loss_fn, optimizer and learning rate scheduler
    2. Checks if saved checkpoint exist, if exists, fetch all states from the checkpoint.
    3. Define inner functions for training and validation
    4. Run main loop for training and validation and save best accurate model as checkpoint
    5. Print the stats for each epoch
    6. Print the training finished message
    7. Print a complete table for stats comparison for loss and acc over train and validate
    8. Plot 2-in-1 plots for loss and accuracy.
    """

    #define optimizer and lr scheduler
    optimizer = optim.AdamW(filter(lambda param: param.requires_grad, model.parameters()),
                            lr=TrainConfig.lr,
                            weight_decay=TrainConfig.decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                        T_max=TrainConfig.t_max,
                                                        eta_min=TrainConfig.eta_min)

    #===============================================
    checkpoint_dir = PATH.CHECKPOINT_DIR
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path_load = PATH.CHECKPOINT_PATH_FOR_LOAD

    if os.path.exists(checkpoint_path_load):
        myLog("Checkpoint found. Resume training (Fine Tuning)")
        #Phase-1:
        # start_epoch, best_valid_acc = utils.load_checkpoint(checkpoint_path_load, model, optimizer, lr_scheduler)
        #Phase-2:
        best_valid_acc = utils.load_model_weights_only(checkpoint_path_load, model)
        start_epoch = 1
    else:
        myLog("No checkpoint found. Starting fresh training (FT)")
        start_epoch = 1
        best_valid_acc = 0

    #If previous loop finished completely, reset start point
    # if start_epoch == TrainConfig.epochs_count + 1:
    #     start_epoch = 1
    #===============================================

    def train(image, label):
        model.train()
        pred = model(image)
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return pred, loss

    def validate(image, label):
        model.eval()
        with torch.inference_mode():
            pred = model(image)
            loss = loss_fn(pred, label)
            return pred, loss

    #=================================================
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    start_time = datetime.now()
    myLog("TRAINING STARTS HERE at "+ start_time.strftime("%d-%m-%Y_%H-%M-%S"))    
    #=================================================
    for epoch in range(start_epoch, TrainConfig.epochs_count+1):

        train_loss = 0
        batch_count = 0
        train_acc = 0
        correct_pred = 0
        total_item = 0
        for image, label in train_loader:
            pred, loss = train(image, label)
            train_loss += loss.item()
            batch_count += 1
            total_item += len(label)
            correct_pred += (label == pred.argmax(dim=-1)).sum().item()

        train_loss /= batch_count
        train_acc = correct_pred*100/total_item
        training_losses.append(train_loss)
        training_accuracies.append(train_acc)

        #====================================
        lr_scheduler.step()
        #====================================

        valid_loss = 0
        batch_count = 0
        valid_acc = 0
        correct_pred = 0
        total_item = 0
        for image, label in valid_loader:
            pred, loss = validate(image, label)  
            valid_loss += loss.item()
            batch_count += 1
            total_item += len(label)
            correct_pred += (label == pred.argmax(dim=-1)).sum().item()

        valid_loss /= batch_count
        valid_acc = correct_pred*100/total_item
        validation_losses.append(valid_loss)
        validation_accuracies.append(valid_acc)
        #=======================================
        checkpoint_path_save = PATH.CHECKPOINT_PATH_FOR_SAVE
        #Save the best accurate model so far
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            utils.save_checkpoint(epoch, model, optimizer, lr_scheduler, best_valid_acc, checkpoint_path_save, checkpoint_dir)

        #=======================================
        addLine()
        print(f"Epoch : {epoch}")
        print(f"Losses :  Train={train_loss:^10.3f}, Validation={valid_loss:^10.3f}")
        print(f"Accuracy: Train={train_acc:^10.3f}, Validation={valid_acc:^10.3f}")
        newLine()

    #===========================================
    end_time = datetime.now()
    myLog("TRAINING ENDS HERE at " + end_time.strftime("%d-%m-%Y_%H-%M-%S"))
    print("Total training duration recorded : ", (end_time - start_time))

    #===========================================
    myLog("Loss and Validation Comparison Table")
    utils.print_comparison_table(train_loss=training_losses, train_acc=training_accuracies,
                                 val_loss=validation_losses, val_acc=validation_accuracies)

    #===========================================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_dir = PATH.PLOT_DIR
    loss_plot_name = f"{PATH.LOSS_PLOT_NAME}_{timestamp}.png"
    acc_plot_name = f"{PATH.ACC_PLOT_NAME}_{timestamp}.png"
    #### Training completed ####
    utils.plot_train_vs_validation_curve(PLOTS_DIR=plot_dir, 
                                         plot_name=loss_plot_name, 
                                         first_item=training_losses,
                                         first_label="Training Loss", 
                                         second_item=validation_losses,
                                         second_label="Validation Loss",
                                         xlabel="Epochs", ylabel="Loss")

    utils.plot_train_vs_validation_curve(PLOTS_DIR= plot_dir,
                                         plot_name=acc_plot_name,
                                         first_item=training_accuracies,
                                         first_label="Training Accuracy",
                                         second_item=validation_accuracies,
                                         second_label="Validation Accuracy",
                                         xlabel="Epochs",
                                         ylabel="Accuracy")