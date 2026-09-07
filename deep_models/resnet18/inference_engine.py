from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
import torch
from config import PATH, IMAGE, TrainConfig
import os, sys
from utils import myLog
from PIL import Image
import io

"""
1. Load the model only once on startup and store it in cache and use for all the inferences. 
2. predict gets called by FastAPI and receive byte object
3. converts byte object into PIL and preprocess it.
4. Pass it to model and get prediction.
5. Find softmax and get top K probabilities
6. Construct a dict with label and confidence
7. return the python dict, fast api will handle it
"""

torch.hub.set_dir(PATH.DOWNLOAD_MODEL_PATH)
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, IMAGE.classes)
model.eval()

best_model_load_path = PATH.CHECKPOINT_PATH_FOR_LOAD
if not os.path.exists(best_model_load_path):
    myLog("No checkpoint found. Abort Testing")
    sys.exit(1)

myLog("Checkpoint found. Using Model for Inference")
checkpoint = torch.load(best_model_load_path)
model.load_state_dict(checkpoint["model_state_dict"])
myLog("MODEL LOADED")
#=====================================================

def predict(input_bytes):
    #1. Convert the bytes in PIL
    pil_image = Image.open(io.BytesIO(input_bytes))

    #2. preprocess the image
    image = preprocess(pil_image)

    #3. add the extra dim for making a batch of size 1
    image = torch.unsqueeze(input=image, dim=0)

    #4. feed input to model and get prediction
    with torch.inference_mode():
        pred = model(image)

    #5. Squeeze the extra dim (batch dim) from pred.
    pred = torch.squeeze(input=pred, dim=0)

    #6. convert raw logits into probabilities
    prob = torch.softmax(input=pred, dim=0)

    #7.Fetch top K prob
    topk_probs, topk_index = torch.topk(prob, k=3)

    #8. convert tensors to list
    topk_probs = topk_probs.tolist()
    topk_index = topk_index.tolist()

    #9. Fetch top K food class names
    topk_names = get_food_class(topk_index)

    #10. make dict {"label": "confidence"}
    result_dict = {topk_names[i] : topk_probs[i] for i in range(len(topk_probs))}

    #11. return the dict, FASTAPI will handle it
    return result_dict


def preprocess(image):
    """We don't transform the validation/testing data from augmentation pov, 
    only training data exclusively gets transformed.
    However, we need to resize and convert the testing data as per resnet architecture."""
    test_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(size=(IMAGE.height, IMAGE.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return test_transform(image)

def get_food_class(indices):
    classes =  ['biryani', 'cholebhature', 'dabeli', 'dal', 'dhokla', 
                'dosa', 'jalebi', 'kathiroll', 'kofta', 'naan', 
                'pakora', 'paneer', 'panipuri', 'pavbhaji', 'vadapav']

    return [classes[index] for index in indices]


import dataset
if __name__ == "__main__":
    d = dataset.load_data_from_cache()
    d = d["train"][1]["image"]
    buf = io.BytesIO()
    d.save(buf, "JPEG")
    bytes = buf.getvalue()

    result = predict(bytes)
    print("result = ", result)
    # result =  {'biryani': 0.9999908208847046, 'naan': 5.117015916766832e-06, 'pakora': 1.3194741086408612e-06}