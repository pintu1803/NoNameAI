import torch.nn as nn
import torch.optim as optim

class MNISTModel():

    def __init__(self, config):

        super().__init__()

        self.input_size = config.input_size
        self.hidden_dim1 = config.hidden_dim1
        self.hidden_dim2 = config.hidden_dim2
        self.hidden_dim3 = config.hidden_dim3
        self.dropout = config.dropout
        self.output_size = config.output_size
        self.decay = config.decay
        self.learning_rate = config.learning_rate

        self.deepModel = nn.Sequential(
            nn.Flatten(),

            nn.Linear(self.input_size, self.hidden_dim1),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_dim2, self.hidden_dim3),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_dim3, self.output_size)
        )

    def loss_fn(self):
        return nn.CrossEntropyLoss()

    def optimizer(self):
        return optim.AdamW(self.parameters(), weight_decay=self.decay, lr=self.learning_rate)

    def parameters(self):
        return self.deepModel.parameters()

    def forward(self, input):
        return self.deepModel(input)

    def showModel(self):
        print("Input dim : ", self.input_size)
        print("Hidden dim-1 : ", self.hidden_dim1)
        print("Hidden dim-2 : ", self.hidden_dim2)
        print("Hidden dim-3 : ", self.hidden_dim3)
        print("Output dim : ", self.output_size)
        print("Dropout : ", self.dropout)
        print("Weight decay : ", self.decay)

    