
import torch
from matplotlib import pyplot as plt

#Loss plot
def plot_train_vs_validation_curve(PLOTS_DIR, plot_name, 
                                    first_item, first_label,
                                    second_item, second_label,
                                    xlabel, ylabel):
    #save in plots dir
    PLOTS_DIR.mkdir(exist_ok=True)
    PLOT = PLOTS_DIR / plot_name
    
    plt.plot(first_item, first_label)
    plt.plot(second_item, second_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()

    plt.savefig(PLOT,
                dpi=300,
                bbox_inches="tight")

    plt.show(block=False)
    plt.pause(10)
    plt.close()


#Save model
def save_trained_parameters(model, CHECKPOINT_DIR, MODEL_NAME):
    print("\nMake checkpoints dir")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    MODEL_FULL_PATH = CHECKPOINT_DIR / MODEL_NAME

    print("\nSaving learned weights...")
    torch.save(model.state_dict(), MODEL_FULL_PATH)
    print("Saved learned weights : ", MODEL_FULL_PATH)
