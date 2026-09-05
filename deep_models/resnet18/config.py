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
    DOWNLOADE_MODEL_PATH: str = BASE_DIR / "model"

    DATASET_DIR: str = BASE_DIR / "dataset"
    PLOT_DIR: str = BASE_DIR / "plots"

@dataclass
class IMAGE:
    height: int = 224
    width: int = 224

@dataclass
class TrainConfig:
    train_percent = 0.85
    valid_percent = 0.15
    epochs_count = 20
    batch_size = 32
    test_batch_size = 16