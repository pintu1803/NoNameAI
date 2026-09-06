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

    total_correct_pred = 0
    total_set_size = 0

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

            #calculate for overall
            total_set_size += len(label)
            total_correct_pred += correct

    #Overall accuracy of the model
    acc = (total_correct_pred*100/total_set_size)
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

