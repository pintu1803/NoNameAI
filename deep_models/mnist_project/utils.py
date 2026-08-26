
import torch
from matplotlib import pyplot as plt

#Loss plot
def plot_loss_curve(PLOTS_DIR, plot_name, train_losses, val_losses):
    #save in plots dir
    PLOTS_DIR.mkdir(exist_ok=True)
    PLOT = PLOTS_DIR / plot_name
    
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()

    plt.savefig(PLOT,
                dpi=300,
                bbox_inches="tight")

    plt.show(block=False)
    plt.pause(10)
    plt.close()


#Accuracy plot
def plot_accuracy_curve(PLOTS_DIR, plot_name, train_acc, val_acc):
    #save in plots dir
    PLOTS_DIR.mkdir(exist_ok=True)
    PLOT = PLOTS_DIR / plot_name
    
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()

    plt.savefig(PLOT,
                dpi=300,
                bbox_inches="tight")

    plt.show(block=False)
    plt.pause(10)
    plt.close()


#Save model
def save_model(model, CHECKPOINT_DIR, MODEL_NAME):
    print("\nMake checkpoints dir")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    MODEL_FULL_PATH = CHECKPOINT_DIR / MODEL_NAME

    torch.save(model.state_dict(), MODEL_FULL_PATH)


#Testing methods
#Test loss curve
def plot_testing_loss_curve(PLOTS_DIR, plot_name, test_losses):
    #save in plots dir
    PLOTS_DIR.mkdir(exist_ok=True)
    PLOT = PLOTS_DIR / plot_name
    
    plt.plot(test_losses, label="Test Moving avg Loss")

    plt.xlabel("Batch")
    plt.ylabel("Loss")

    plt.legend()

    plt.savefig(PLOT,
                dpi=300,
                bbox_inches="tight")

    plt.show(block=False)
    plt.pause(10)
    plt.close()

#Test Accuracy plot
def plot_testing_accuracy_curve(PLOTS_DIR, plot_name, test_accuracy):
    #save in plots dir
    PLOTS_DIR.mkdir(exist_ok=True)
    PLOT = PLOTS_DIR / plot_name
    
    plt.plot(test_accuracy, label="Test Moving avg Accuracy")

    plt.xlabel("Batch")
    plt.ylabel("Accuracy")

    plt.legend()

    plt.savefig(PLOT,
                dpi=300,
                bbox_inches="tight")

    plt.show(block=False)
    plt.pause(10)
    plt.close()

#Moving average
def moving_average(data_list, count):
    moving_avg = []

    moving_sum = 0.0
    idx_before_window = 0

    for idx in range(len(data_list)):
        moving_sum += data_list[idx]

        if idx - idx_before_window > count:
            moving_sum -= data_list[idx_before_window]
            avg = moving_sum/count
            moving_avg.append(avg)
            idx_before_window += 1

    return moving_avg



