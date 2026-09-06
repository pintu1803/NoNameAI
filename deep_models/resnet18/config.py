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

    LOSS_PLOT_NAME: str =  "train_vs_validate_loss"
    ACC_PLOT_NAME: str =  "train_vs_validate_acc"

@dataclass
class IMAGE:
    height: int = 224
    width: int = 224
    classes: int = 15

@dataclass
class TrainConfig:
    train_percent: float = 0.85
    valid_percent: float = 0.15
    eta_min: float = 1e-6
    t_max: int = 20
    epochs_count: int = 20
    batch_size: int = 32
    test_batch_size: int = 16
    lr: float = 0.001
    decay: float = 0.002
    