#Test the saved model.

import dataset
from model import cnn_model2
from test import test_model
from config import ModelConfig
from config import PATH
import torch

def main():
    _, _, test_loader = dataset.main()

    config = ModelConfig()

    # model-1 has poor architecture, it's obsolete now.
    # modelObject = cnn_model(config)
    modelObject = cnn_model2(ModelConfig)

    #Training is not required, use saved weights
    model = modelObject.model
    model.load_state_dict(torch.load(PATH.SAVED_MODEL_FULL_PATH))

    test_model(modelObject, test_loader)

###########################
if __name__ == "__main__":
    main()