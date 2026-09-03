
import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    #default stride is 1 for 2nd block, as first block needs stride 2
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()

        #inside a block, first conv uses stride=2, and second uses stride=1
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=3,
                               padding=1,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.shortcut = nn.Identity()

        #if mismatch is there, match the values.
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channels=in_channel,
                                      out_channels=out_channel,
                                      kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)

        identity_projection = self.shortcut(identity)
        identity_projection = self.bn2(identity_projection)

        x = x + identity_projection
        x = self.relu(x)

        return x

###################################################
class ResNet(nn.Module):

    def __init__(self, block, blocks_count, number_of_classes):
        super().__init__()

        #Image net uses 224x224 image, filter of size 7, and padding = 3
        #stem first
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               padding=3, kernel_size=7,
                               stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        #layers now
        input_channel = 64
        output_channel = 64
        self.layer1 = self._make_layer(input_channel, output_channel, block, blocks_count, stride=1)

        input_channel = output_channel
        output_channel = 128
        self.layer2 = self._make_layer(input_channel, output_channel, block, blocks_count)

        input_channel = output_channel
        output_channel = 256
        self.layer3 = self._make_layer(input_channel, output_channel, block, blocks_count)

        input_channel = output_channel
        output_channel = 512
        self.layer4 = self._make_layer(input_channel, output_channel, block, blocks_count)

        #adaptive avg pool
        self.adaptiveAvgPool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=output_channel, out_features=number_of_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxPool(x)

        print("shape after max pool - ", x.shape)

        x = self.layer1(x)
        print("shape after layer1 - ", x.shape)
        x = self.layer2(x)
        print("shape layer2 - ", x.shape)
        x = self.layer3(x)
        print("shape layer3 - ", x.shape)
        x = self.layer4(x)
        print("shape layer4 - ", x.shape)

        x = self.adaptiveAvgPool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

    def _make_layer(self, in_channel, out_channel, block, blocks_count, stride=2):
        blocks = []

        #only first block uses stride=2
        block_first = block(in_channel=in_channel,
                            out_channel=out_channel,
                            stride=stride)

        blocks.append(block_first)

        for _ in range(1, blocks_count):
            blocks.append(block(out_channel,
                                out_channel, 
                                stride=1)
                        )

        return nn.Sequential(*blocks)


###################################################
def ResNet18(number_of_classes):
    resnet18 = ResNet(BasicBlock, 2, number_of_classes)
    return resnet18

def main():
    model = ResNet18(10)

    #created batch of 1 image, with 3 channels and 224 H and W
    x = torch.randn(1, 3, 224, 224)

    y = model(x)

    print("\nPredication tensor : ", y)
    print("\nPredict most probable class : ", torch.argmax(y, dim=-1))

###################################################
if __name__ == "__main__":
    main()