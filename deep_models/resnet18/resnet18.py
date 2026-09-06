
from torchvision.models import resnet18, ResNet18_Weights
import torch
from config import PATH, IMAGE, TrainConfig

def load_resnet18():
    """
    Note: Order of steps matters.
    1. Load the model architecture and pretrainied weights from cache dir.
    2. Replace the FC layer with number of classes required as per dataset.
    Phase-1:
        3. Freeze the complete backbone, except FC layer (will train fc)
        4. Put all BN modules in eval mode. (bn needs stats to be computed)
    Phase-2:
        3. Freeze upto layer-3, unfreeze layer-4 and fc layer
        4. Put all BN modules in eval mode, except for layer-4 BN
    """
    #define path to store model and weights
    torch.hub.set_dir(PATH.DOWNLOAD_MODEL_PATH)
    weight = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weight)
    print("Verify load path : ", torch.hub.get_dir())

    #change the the model classifier
    print("\nFC layer before : ", model.fc)
    model.fc = torch.nn.Linear(model.fc.in_features, IMAGE.classes)
    print("FC layer after : ", model.fc)

    #=====================================================
    # #Phase-1: Freeze the complete backbone
    # for param in model.parameters():
    #     param.requires_grad = False

    # #Phase-1: Put the BN modules in eval mode
    # for module in model.modules():
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         module.eval()
    
    #-----------------------------------------------------

    #Phase-2: Unfreeze layer4 and FC layer
    #They learn more high-level features compared to earlier layers
    for name, layer in model.named_parameters():
        if "layer4" in name:
            layer.requires_grad = True
        else:
            layer.requires_grad = False

    #Phase-2: Put the BN modules in eval mode, except for layer4
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d) and "layer4" not in name:
            module.eval()
    #=====================================================

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