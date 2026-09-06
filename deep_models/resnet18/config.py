import os
from pathlib import Path
from dataclasses import dataclass

#Gobal code gets executed when we import this file.

print("Current workig dir : ",os.getcwd())
BASE_DIR = Path(__file__).resolve().parent
print("Project Base dir : ", BASE_DIR)


@dataclass
class PATH:
    CHECKPOINT_DIR: str = BASE_DIR / "checkpoints"
    CHECKPOINT_PATH: str = CHECKPOINT_DIR / "best_val_acc_model.pth"
    DOWNLOADE_MODEL_PATH: str = BASE_DIR / "model"

    DATASET_DIR: str = BASE_DIR / "dataset"
    PLOT_DIR: str = BASE_DIR / "plots"

    LOSS_PLOT_NAME: str = PLOT_DIR / ""
    ACC_PLOT_NAME: str = PLOT_DIR / ""

@dataclass
class IMAGE:
    height: int = 224
    width: int = 224
    classes: int = 15

@dataclass
class TrainConfig:
    train_percent: float = 0.85
    valid_percent: float = 0.15
    epochs_count: int = 20
    batch_size: int = 32
    test_batch_size: int = 16
    lr: float = 0.001
    decay: float = 0.002