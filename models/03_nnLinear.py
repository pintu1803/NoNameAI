import torch
import torch.nn as nn
import torch.optim as optim

"""
Linear shape = (input_features, output_features)
Linear(2,3) -> creates a linear layer for accepting input with 2 features, and output with 3 features
It generates 2 learnable parameters W and b -> Weights and bias
Weight matrix has shape (3,2) -> just opposite of Linear dimensions.
Because torch does this computation: y = x @ W.T + b
x shape = (batch_size, input_features)
w shape = (output_features, input_features)
w.T shape = (input_features, output_features)
x @ w.T = (batch_size, input_features) @ (input_features, output_features) = (batch_size, output_features)

input(batch_size, input_features) -> Model -> output(batch_size, output_features)
"""

### why different modules for loss and optimizers ?

"""
Model, Activations and Loss function -> all are involved in forward pass
they are components used in forward pass, so they all are kept together in nn module.

Optimizers come into picture after the gradients are computed.
They are kept in separate module -> torch.optim Examples: SGD and Adam.
They only take parameters as the input who have the gradients. 
"""

### why are we calling model(x) whereas model is an object of nn.Linear class

"""
nn.Linear is a class, and model = nn.Linear(x, y) gives an object -> model
then how does model(X) is called ? This is made possible by dunden method
__call__() in python's object oriented programming.
it makes an object callable and behind the scene __call__() method runs.

But, here we are not supposed to call model.__call__(self, x) method itself,
because __call__() runs extra stuff which we would skip if we run call directly.
"""
# Dataset
X = torch.tensor([
    [1.,2.],
    [2.,1.],
    [3.,2.],
    [4.,3.]
])

# Labels
Y = torch.tensor([
    [5.],
    [4.],
    [7.],
    [10.]
])

# Creates W=(1,2) and b=(2,)
model = nn.Linear(2,1)
#what is model?
print("what is model ? ", type(model))

# creates mean squared error loss function
criterion = nn.MSELoss()

# creates stochastic gradient descent function
# it updates all learnable parameters by the formula
# w = w - lr * derivative
optimizer = optim.SGD(
    model.parameters(),
    lr=0.05
)

# Initial weight and bias
print("\nInitial weight:", model.weight)
print("\nInitial bias:", model.bias)
print()

# Run 5 epochs
for epoch in range(5):

    print(f"\nEpoch {epoch+1}")

    prediction = model(X)

    loss = criterion(prediction, Y)

    optimizer.zero_grad()

    loss.backward()

    print("\nBefore update, weights : ", model.weight.data)
    print("Before update, bias : ", model.bias.data)

    optimizer.step()
    
    print("\nLoss:", loss.item())
    print("After update, weights : ", model.weight.data)
    print("After update, bias : ", model.bias.data)
    print()


### Two linear layers in sequence
"""
If we stack two Linear layers, then the resultant is also a Linear layer.
y = wx + b -> this represents a straight line and the model can learn simple line relationship perfectly.
But, what if y = x ^ 2 or y = x ^ 3. Then single Linear layer or multiples stacked together won't be helpful.
h = w1x+b1
y = w2h+b2
-----------
y = w2(w1x + b1) + b2
y = w1.w2.x + (b1.w2 + b2)
This is just a bigger linear equation, and can't learn y = x^2 behaviour.
"""

### Get the terminologies correct
"""
layer1 = nn.Linear(3,4)
layer2 = nn.Linear(4,2)

Each Linear layer consistes of 2 learnable tensors/parameters -> weight and bias.
so, here for two layers, we have -> 2 weight tensors(w1, w2), and 2 bias tensors(b1, b2)
And, w1 has 12 numbers/weights, w2 has 8 weights, b1 has 4 weights and b2 has 2 weights.

Terminologies: Learnable parameters means learnable tensors.
weights/numbers means what makes tensors.
"""


