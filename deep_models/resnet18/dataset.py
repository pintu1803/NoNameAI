
import datasets
from config import PATH
from datasets import load_dataset
from utils import myLog, addLine

myLog("WELCOME TO RESNET18 FINE TUNING")

print("\nLoad the dataset from cache dir : ", PATH.DATASET_DIR)
dataset = load_dataset("bharat-raghunathan/indian-foods-dataset",
                       cache_dir=PATH.DATASET_DIR)

"""
dataset : DatasetDict({
    train: Dataset({
        features: ['image', 'label'],
        num_rows: 3809
    })
    test: Dataset({
        features: ['image', 'label'],
        num_rows: 961
    })
})

dataset is a dict, so key-indexing are allowed.
train is a dataset (list of items), so indexing works to fetch record - which is a tuple (image, label)
features and num_rows are instance parameters, access them in similar way
features are dict and can be accessed using keys - image and label
"""

print("\nDataset loaded.. Begin inspection")
print("\nWhat is our dataset?")
print("\nType: ", type(dataset))
print("\ndataset : ", dataset)

train = dataset["train"]
test = dataset["test"]
print("\nSplit the two parts of the dict. Inspect their types and internal structures")
print("\nType : ", type(train))
print("\nTrain set : ", train)
print("\nType : ", type(test))
print("\nTest set :", test)
addLine()

msg = f"\nTrain and Test are Dataset Objects unlike dataset which is a DefaultDict\n"
print(msg)

#Explore internals of train
try:
    train_sample = train[0]
    print("Train sample, using index-0 : ", train_sample)
except Exception as e:
    print("exp : ", e)

#Explore internals recursively
try:
    train_features = train.features
    print("\ntrain_features = ", train_features)
    train_num_rows = train.num_rows
    print("\nTrain num rows : ", train_num_rows)
    train_features_imageKey = train_features['image']
    print("\train_features_imageKey : ", train_features_imageKey)
    train_features_labelKey = train_features['label']
    print("\ntrain_features_labelKey : ", train_features_labelKey, type(train_features_labelKey))
except Exception as e:
    print("exp : ", e)

addLine()
##########################################################