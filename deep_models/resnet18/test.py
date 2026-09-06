
import os
import sys
from utils import myLog, addLine
import utils
import torch
from config import PATH
from datetime import datetime

def test_model(test_loader, model, best_model_load_path, loss_fn):

    if not os.path.exists(best_model_load_path):
        myLog("No checkpoint found. Abort Testing")
        sys.exit(1)

    myLog("Checkpoint found. Using Model for Inference")
    utils.load_model_weights_only(best_model_load_path, model)

    test_batch_accuracies = []
    test_batch_losses = []
    #==========================================
    model.eval()
    with torch.inference_mode():
        total_correct_pred = 0
        total_set_size = 0

        for idx, (image, label) in enumerate(test_loader):
            pred = model(image)
            loss = loss_fn(pred, label)

            #per batch metrics
            test_loss = loss.item()
            correct_batch_pred = (label == pred.argmax(dim=-1)).sum().item()
            test_acc = correct_batch_pred*100/len(label)

            total_correct_pred += correct_batch_pred
            total_set_size += len(label)

            #append per batch results in the lists
            test_batch_losses.append(test_loss)
            test_batch_accuracies.append(test_acc)

            #=======================================
            addLine()
            print(f"Batch : {idx+1}")
            print(f"Loss = {test_loss:.3f}, Accuracy = {test_acc:.3f}%")

    #================================================
    total_acc = total_correct_pred*100/total_set_size
    myLog(f"Total test accuracy = {total_acc:.3f} %")

    total_avg_loss = sum(test_batch_losses)/len(test_batch_losses)
    myLog(f"Net avg test loss = {total_avg_loss:^10.3f}")
    addLine()
    #================================================

    PLOTS_DIR = PATH.PLOT_DIR
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOSS_PLOT_NAME = PLOTS_DIR / f"Test_loss_curve_{timestamp}.png"
    ACC_PLOT_NAME = PLOTS_DIR / f"Test_acc_curve_{timestamp}.png"

    #Plot the loss and accuracy curves
    utils.plot_testing_curve(PLOTS_DIR, LOSS_PLOT_NAME, test_batch_losses, "Testing Loss", "Batch", "Loss")
    utils.plot_testing_curve(PLOTS_DIR, ACC_PLOT_NAME, test_batch_accuracies, f"Testing Accuracy, overall {total_acc:.2f}%", "Batch", "Accuracy")