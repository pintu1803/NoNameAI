
import torch
import torch.nn as nn

### dataset has negative and positive input which enables model to capture whole parabolic nature
### negative output of Linear layer which becomes input to ReLU is what breaks "big linear layer" curse
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


"""
Interpolation: Predicting within the training domain [-1, 1]
Extrapolation: Predicting outside the training domain [-2, 2]
NN behaves like linear model outside trainig domain.
Model predicts accurately within [-1, 1] but shows fixed linear behaviour in range [-2, -1] and [1, 2]
"""

#Interpolation
X = torch.linspace(-1, 1, 20).reshape(20, 1)
Y = X ** 2

with torch.no_grad():
    test_pred = model(X)

print("\nCompare pred and labels from unseen testing data")
for i in range(20):
    print(f"X={X[i].item() : .2f} "
          f"actual={Y[i].item() : .4f} "
          f"pred={test_pred[i].item() : .4f} "
    )

#Extrapolation
X = torch.linspace(-2, 2, 20).reshape(20, 1)
Y = X ** 2

with torch.no_grad():
    test_pred = model(X)

print("\nCompare pred and labels from unseen testing data")
for i in range(20):
    print(f"X={X[i].item() : .2f} "
          f"actual={Y[i].item() : .4f} "
          f"pred={test_pred[i].item() : .4f} "
    )