
import torch.nn as nn

class cnn_model():

    def __init__(self, tc):
        self.image_height = tc.image_height
        self.image_width = tc.image_width
        self.kernel_size = tc.kernel_size
        self.in_channel = tc.in_channel
        self.out_channel1 = tc.out_channel1
        self.out_channel2 = tc.out_channel2
        self.out_channel3 = tc.out_channel3
        self.linear_hidden_dim = tc.linear_hidden_dim
        self.padding = tc.padding
        self.stride = tc.stride
        self.max_pool_size = tc.max_pool_size
        self.classification_count = tc.classification_count

        self.model = nn.Sequential(
            #layer-1
            nn.Conv2d(in_channels=self.in_channel, 
                      out_channels=self.out_channel1, 
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding),
            nn.ReLU(),
            # nn.MaxPool2d(self.max_pool_size),

            #layer-2
            nn.Conv2d(in_channels=self.out_channel1, 
                      out_channels=self.out_channel2, 
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(self.max_pool_size),

            #layer-3
            nn.Conv2d(in_channels=self.out_channel2, 
                      out_channels=self.out_channel3, 
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(self.max_pool_size),

            #layer-4
            nn.Flatten(),
            nn.Linear(self.linear_layer1_input(), self.linear_hidden_dim),
            nn.ReLU(),

            #layer-5
            nn.Linear(self.linear_hidden_dim, self.classification_count)
        )

    def linear_layer1_input(self):
        # return self.out_channel3 * final_height_after_2d_pool * final_width_after_pool
        # return 16 * self.out_channel3
        return 64 * self.out_channel3

    def forward(self, input):
        return self.model(input)
    