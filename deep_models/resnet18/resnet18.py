
from torchvision.models import resnet18, ResNet18_Weights
import torch
from config import PATH, IMAGE, TrainConfig

def load_resnet18():
    """
    Note: Order of steps matters.
    1. Load the model architecture and pretrainied weights from cache dir.
    2. Freeze the backbone
    3. Replace the FC layer with number of classes required as per dataset.
    4. Put all BN modules in eval mode.
    6. Else, continue with fresh model.
    """
    #define path to store model and weights
    torch.hub.set_dir(PATH.DOWNLOADE_MODEL_PATH)
    weight = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weight)
    print("Verify load path : ", torch.hub.get_dir())

    #Freeze the backbone
    for param in model.parameters():
        param.requires_grad = False

    #change the the model classifier
    print("\nFC layer before : ", model.fc)
    model.fc = torch.nn.Linear(model.fc.in_features, IMAGE.classes)
    print("FC layer after : ", model.fc)

    #Put the BN modules in eval mode
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()

    #Return the model
    return model

def inspect_model(model):
    #Lets inspect model.
    print("\nModel ", model._get_name())

    total_parameters = sum(p.numel() for p in model.parameters())
    print("\nTotal parameters : ", total_parameters)

    total_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nTotal learnable parameters : ", total_learnable_parameters)

    # print("Model architecture : ", model)
    # print("Model layer 1 : ", model.layer1)
    # print("Model layer 2 : ", model.layer2)
    # print("Model layer 3 : ", model.layer3)
    # print("Model layer 4 : ", model.layer4)
    # print("Model Fully Connected layer : ", model.fc)

    print("\nModel layer1[0] : ", model.layer1[0])

    # for name, layer in model.named_children():
    #     print("(name, layer) : ", name, layer)

    # for name, module in model.named_modules():
    #     print("(name, module) : ", name, module)
    #     break

    # for name, member in model._named_members():
    #     print("(name, member) : ", name, member)

    list_of_tensors = list(model.parameters())
    num_tensors = len(list(model.parameters()))

    print("\nType of one tensor and shape = ", type(list_of_tensors[0]), list_of_tensors[0].shape)
    print("\nNumber of tensors = ", num_tensors)

    print("\nModel training status : ", model.training)

    layers = len(list(model.modules()))
    print("\nNumber of layers in the model : ", layers)

    relu_list = [layer for layer, name in model.named_modules() if 'relu' in layer]
    print("\nNumber of relu activation : ", len(relu_list))

######################
def main():
    model = load_resnet18()
    inspect_model(model)

if __name__ == "__main__":
    main()