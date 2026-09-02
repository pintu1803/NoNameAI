import os
from pathlib import Path
from dataclasses import dataclass

#Gobal code gets executed when we import this file.

print("Current workig dir : ",os.getcwd())
BASE_DIR = Path(__file__).resolve().parent
print("Project Base dir : ", BASE_DIR)

@dataclass
class TrainConfig:
    normMean: float = 0.5
    normVar: float = 0.5
    train_percent: float = 0.8
    valid_percent: float = 0.2
    epoch_count: int = 10
    epoch_checkpoint: int = 1
    batch_size: int = 64
    shuffle_train_set: bool = True
    drop_last_in_train_set: bool = True

    learning_rate: float = 0.005
    weigth_decay: float = 0.01


@dataclass
class ModelConfig:
    image_height: int = 32
    image_width: int = 32
    kernel_size: int = 3
    in_channel: int = 3
    out_channel1: int = 32
    out_channel2: int = 64
    out_channel3: int = 128
    linear_hidden_dim: int = 256
    padding: int = 1
    stride: int = 1
    max_pool_size: int = 2 
    classification_count: int = 10

@dataclass
class PATH:
    CHECKPOINT_DIR: str = BASE_DIR / "checkpoints"
    SAVED_MODEL_FULL_PATH: str = CHECKPOINT_DIR / "cifar10_cnn_model_83_acc.pth"
    SAVE_MODEL_WITH_NAME: str = "cifar10_cnn_model.pth"

    TRAIN_DATA_DIR: str = BASE_DIR / "train_data"
    TEST_DATA_DIR: str = BASE_DIR / "test_data"
    PLOT_DIR: str = BASE_DIR / "plots"