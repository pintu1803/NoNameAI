from dataclasses import dataclass

import os
from pathlib import Path

#Gobal code gets executed when we import this file.

print("Current workig dir : ",os.getcwd())
BASE_DIR = Path(__file__).resolve().parent
print("Project Base dir : ", BASE_DIR)

@dataclass
class ModelConfig:
    input_size = 784
    hidden_dim1 = 256
    hidden_dim2 = 256
    hidden_dim3 = 256
    output_size = 10
    dropout = 0.2
    decay = 0.01
    learning_rate = 0.005

@dataclass
class TrainConfig:
    epoch_count = 10
    epoch_checkpoint = 1

@dataclass
class DatasetConfig:
    downloadTrain = False
    downloadTest = False