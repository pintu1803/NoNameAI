import os
from pathlib import Path
from dataclasses import dataclass

#Gobal code gets executed when we import this file.

print("Current workig dir : ",os.getcwd())
BASE_DIR = Path(__file__).resolve().parent
print("Project Base dir : ", BASE_DIR)

@dataclass
class TrainConfig:
    train_percent = 0.8
    valid_percent = 0.2
    epoch_count: int = 20
    epoch_checkpoint: int = 1
    batch_size = 64
    shuffle_train_set = True
    drop_last_in_train_set = True