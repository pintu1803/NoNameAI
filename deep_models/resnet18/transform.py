from torchvision import transforms
from config import IMAGE
import utils

def preprocessing_augmentation_transform():
    """
    First of all, convert the image into RGB, if >3 channels exist, drop them.
    Mean and std are fixed and come from the original imagenet paper
    """
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(size=(IMAGE.height, IMAGE.width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                               hue=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    utils.myLog("Returning transform for preprocessing + augmenting training data")
    return train_transform


def preprocessing_transform():
    """We don't transform the validation/testing data from augmentation pov, 
    only training data exclusively gets transformed.
    However, we need to resize and convert the testing data as per resnet architecture."""
    test_transform = transforms.Compose([
        transforms.Resize(size=(IMAGE.height, IMAGE.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    utils.myLog("Returning transform for preprocessing validation/testing data")
    return test_transform

#########################
def main():
    preprocessing_augmentation_transform()
    preprocessing_transform()

if __name__ == "__main__":
    main()
