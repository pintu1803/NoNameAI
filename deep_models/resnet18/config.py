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

    TRAIN_DATA_DIR: str = BASE_DIR / "train_data"
    TEST_DATA_DIR: str = BASE_DIR / "test_data"
    PLOT_DIR: str = BASE_DIR / "plots"