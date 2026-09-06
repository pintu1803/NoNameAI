
from matplotlib import pyplot as plt
import torch

def myLog(title):
    margin = "*" * ((78 - len(title))//2)
    print("\n")
    print("=" * 80)
    print(f"{margin} {title} {margin}")
    print("=" * 80)

def addLine():
    # print("\n")
    print("=" * 80)
    # print("\n")

def newLine():
    print("\n")

def showImage(sample, label, time=2):
    plt.imshow(sample["image"])
    plt.title(label)

    plt.axis("off")
    plt.show(block=False) #don't pause code execution here, keep going on
    plt.pause(time)
    plt.close()

#save model checkpoint
def save_checkpoint(epoch, model, optimizer, scheduler, best_valid_acc, MODEL_CHECKPOINT_PATH):
    torch.save({"epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_valid_acc": best_valid_acc},
                MODEL_CHECKPOINT_PATH)
    
#load saved model configs
def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["start_epoch"]
    best_valid_acc = checkpoint["best_valid_acc"]
    return start_epoch, best_valid_acc

#plot two curves - train vs validation
def plot_train_vs_validation_curve(PLOTS_DIR, plot_name, 
                                    first_item, first_label,
                                    second_item, second_label,
                                    xlabel, ylabel):
    #save in plots dir
    PLOTS_DIR.mkdir(exist_ok=True)
    PLOT = PLOTS_DIR / plot_name
    
    plt.plot(first_item, label=first_label)
    plt.plot(second_item, label=second_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()

    plt.savefig(PLOT,
                dpi=300,
                bbox_inches="tight")

    plt.show(block=False)
    plt.pause(10)
    plt.close()

#print comparison table
def print_comparison_table(train_loss, train_acc, val_loss, val_acc):

    print("_"*65)
    print("| train_losses | train_accuracies | val_losses | val_accuracies |")
    for i in range(len(train_loss)):
        print(f"| {train_loss[i] : ^12.2f} | {train_acc[i] : ^16.2f} | {val_loss[i] : ^10.2f} | {val_acc[i] : ^14.2f} |")
    print("*"*65)
