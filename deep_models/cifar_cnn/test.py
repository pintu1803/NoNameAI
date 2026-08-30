##Contains absolute method, no method

from config import BASE_DIR
import utils
import torch
import torch.nn as nn

def test_model(modelObj, test_loader):

    modelObj.model.eval()
    loss_fn = nn.CrossEntropyLoss()
    #TEST the model
    print("\nTESTING TIME\n")

    test_losses = []
    test_accuracy = []

    with torch.inference_mode():
        for idx, (image, label) in enumerate(test_loader):
            pred = modelObj.forward(image)
            loss = loss_fn(pred, label)  

            #calculate loss and acc for this batch
            total = len(label)
            correct += (label == pred.argmax(dim=-1)).sum()
            accuracy = correct*100/total

            #Store loss and acc
            test_losses.append(loss.item())
            test_accuracy.append(accuracy)

    #inference mode ends here
    moving_avg_loss = utils.moving_average(test_losses, 100)
    moving_avg_acc = utils.moving_average(test_accuracy, 100)

    #Prepare dir and plot names
    PLOTS_DIR = BASE_DIR / "plots"
    LOSS_PLOT_NAME = PLOTS_DIR / "Test_loss_curve.png"
    ACC_PLOT_NAME = PLOTS_DIR / "Test_acc_curve.png"

    #Plot the loss and accuracy curves
    utils.plot_testing_loss_curve(PLOTS_DIR, LOSS_PLOT_NAME, moving_avg_loss)
    utils.plot_testing_accuracy_curve(PLOTS_DIR, ACC_PLOT_NAME, moving_avg_acc)
