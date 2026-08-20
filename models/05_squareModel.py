
import torch
import torch.nn as nn


### This dataset has all positive training dataset
### which makes ReLU useless and model does not learn non-linearity
# x = torch.arange(100, dtype=torch.float32).reshape(100, 1)
# x = x/100
# y = x ** 2


### dataset has negative input which produces negative output in linear layers
### negative input to ReLU is what breaks "big linear layer" curse
x = torch.linspace(-1, 1, 100).reshape(100, 1)
y = x ** 2

### size 8 also works but slightly underfit.
### size 32 also works as it includes almost 4x weights
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

print("\nSequential model is : ", model)
print("Loss criterion is : ", criterion)
# print("Optimizer is : ", optimizer)

for epoch in range(5000):
    pred = model(x)

    loss = criterion(pred, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if(epoch % 500 == 0):
        print("Loss computed - Epoch : {epoch}: ", loss.item())


print("Now lets predict")

### Using some random portion of the trained data
### to predict and compare with original labels
with torch.no_grad():
    pred = model(x)

print("\nCompare pred and labels from original dataset")
for i in range(1,100,5):
    print(f"x={x[i].item() : .2f} "
          f"actual={y[i].item() : .4f} "
          f"pred={pred[i].item() : .4f} "
    )


### Predict output for real unseen data
### It follows similar pattern of training dataset
X = torch.linspace(-1, 1, 20).reshape(20, 1)
Y = X ** 2
test_pred = model(X)

print("\nCompare pred and labels from unseen testing data")
for i in range(20):
    print(f"X={X[i].item() : .2f} "
          f"actual={Y[i].item() : .4f} "
          f"pred={test_pred[i].item() : .4f} "
    )

