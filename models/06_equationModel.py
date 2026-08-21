
import torch
import torch.nn as nn

"""
Learn y = x^3 + x^2 + x + 1
Next we will learn [x, y] -> [x^2 - y^2]
"""

### Prepare dataset
x = torch.linspace(0, 5, 1000).reshape(1000, 1)
y = x ** 3 + x ** 2 + x + 1

### Split dataset into training, validation and testing
x_train = x[:800]
y_train = y[:800]

x_val = x[800:900]
y_val = y[800:900]

x_test = x[900:]
y_test = y[900:]

### Sequential Model
hidden_dim = 128
model = nn.Sequential(
    nn.Linear(1, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 1)
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

print("\nSequential model's hidden linear layer is : ", model[2])
print("Loss criterion is : ", criterion)


### Start training here
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