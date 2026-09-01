##Contains absolute method, no method

from config import PATH
import utils
import torch
import torch.nn as nn
from datetime import datetime

def test_model(modelObj, test_loader):

    modelObj.model.eval()
    loss_fn = nn.CrossEntropyLoss()
    #TEST the model
    print("\nTESTING TIME\n")

    test_losses = []
    test_accuracy = []

    with torch.inference_mode():
        for idx, (image, label) in enumerate(test_loader):
            pred = modelObj.model(image)
            loss = loss_fn(pred, label)  

            #calculate loss and acc for this batch
            total = len(label)
            correct = (label == pred.argmax(dim=-1)).sum()
            accuracy = correct*100/total

            #Store loss and acc
            test_losses.append(loss.item())
            test_accuracy.append(accuracy)

    #Overall accuracy of the model
    acc = (sum(test_accuracy)/len(test_accuracy)).item()
    print(f"Overall testing accuracy : {acc : .3f} %")

    #inference mode ends here
    moving_avg_loss = utils.moving_average(test_losses, 100)
    moving_avg_acc = utils.moving_average(test_accuracy, 100)

    #Prepare dir and plot names
    PLOTS_DIR = PATH.PLOT_DIR
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOSS_PLOT_NAME = PLOTS_DIR / f"Test_loss_curve_{timestamp}.png"
    ACC_PLOT_NAME = PLOTS_DIR / f"Test_acc_curve_{timestamp}.png"

    #Plot the loss and accuracy curves
    utils.plot_testing_curve_with_moving_avg(PLOTS_DIR, LOSS_PLOT_NAME, 
                                            moving_avg_loss, "Testing Loss",
                                            "Batch", "Loss")
    
    utils.plot_testing_curve_with_moving_avg(PLOTS_DIR, ACC_PLOT_NAME, 
                                             moving_avg_acc, "Testing Accuracy",
                                             "Batch", "Loss")

