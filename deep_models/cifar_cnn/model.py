
import torch.nn as nn
import math

class cnn_model2(nn.Module):

    def __init__(self, tc):
        super().__init__()
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
            nn.Conv2d(in_channels=self.out_channel1, 
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
            nn.Conv2d(in_channels=self.out_channel2, 
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
            nn.Conv2d(in_channels=self.out_channel3, 
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
        def conv_out(image_size):
            return math.floor((image_size + 2*self.padding - self.kernel_size) / self.stride) + 1

        def pool_out(image_size):
            return math.floor(image_size/self.max_pool_size)

        #conv layer-1
        h1 = conv_out(self.image_height)
        w1 = conv_out(self.image_width)

        #conv layer-2
        h2 = conv_out(h1)
        w2 = conv_out(w1)

        h2 = pool_out(h2)
        w2 = pool_out(w2)

        #conv layer-3
        h3 = conv_out(h2)
        w3 = conv_out(w2)

        h3 = pool_out(h3)
        w3 = pool_out(w3)

        return self.out_channel3 * w3 * h3
    
class cnn_model1(nn.Module):

    def __init__(self, tc):
        super().__init__()
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
        def conv_out(image_size):
            return math.floor((image_size + 2*self.padding - self.kernel_size) / self.stride) + 1

        def pool_out(image_size):
            return math.floor(image_size/self.max_pool_size)

        #conv layer-1
        h1 = conv_out(self.image_height)
        w1 = conv_out(self.image_width)

        #conv layer-2
        h2 = conv_out(h1)
        w2 = conv_out(w1)

        h2 = pool_out(h2)
        w2 = pool_out(w2)

        #conv layer-3
        h3 = conv_out(h2)
        w3 = conv_out(w2)

        h3 = pool_out(h3)
        w3 = pool_out(w3)

        return self.out_channel3 * w3 * h3
