
import torch
import torch.nn as nn

"""
Learn y = x^3 + x^2 + x + 1
Next we will learn [x, y] -> [x^2 - y^2]
"""

### Function
def make_labels(X):
    Y = X ** 3 + X ** 2 + X + 1
    return Y

### Prepare dataset and shuffle it
X = torch.linspace(0, 5, 1000).reshape(1000, 1)
Y = make_labels(X)
perm = torch.randperm(1000)
x = X[perm]
y = Y[perm]


### without shuffling splitting is like extrapolation
### Split dataset into training, validation and testing
x_train = x[:800]
y_train = y[:800]

x_val = x[800:900]
y_val = y[800:900]

x_test = x[900:]
y_test = y[900:]

### Sequential Model
#256 worked fine
#128 worked better than 256 for testing data
#64 too worked just fine
#32 fine
#16 fine
# 8 shows significantly wrong predictions
"""
x= 0.48 y_test= 1.8090 y_predicted= 1.7738
x= 2.46 y_test= 24.4579 y_predicted= 24.5316
x= 1.45 y_test= 7.6160 y_predicted= 8.6918
x= 3.05 y_test= 41.8321 y_predicted= 43.5741
x= 4.99 y_test= 155.1407 y_predicted= 150.6515
x= 3.70 y_test= 69.2264 y_predicted= 66.0359
x= 4.94 y_test= 151.3136 y_predicted= 147.6883
x= 4.10 y_test= 90.7773 y_predicted= 92.0462
x= 4.50 y_test= 117.1940 y_predicted= 118.7149
x= 4.46 y_test= 114.3791 y_predicted= 116.0810
"""
hidden_dim = 8
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
    # model.train() # add this when our model has Dropout and Batch normalisation

    pred_y_train = model(x_train)

    loss_train = criterion(pred_y_train, y_train)

    optimizer.zero_grad()

    loss_train.backward()

    optimizer.step()

    if(epoch % 500 == 0):
        print(f"Epoch : {epoch} Training loss computed : {loss_train.item() : .4f}")


### Validation step
# model.eval() # add this when our model has Dropout and Batch normalisation

with torch.inference_mode():
    pred_y_val = model(x_val)
    loss_val = criterion(pred_y_val, y_val)
    print(f"\nValidation loss computed :  {loss_val.item() : .4f}")


### Testing step
with torch.inference_mode():
    pred_y_test = model(x_test)
    loss_test = criterion(pred_y_test, y_test)
    print(f"\nTesting loss computed : {loss_test.item() : .4f}\n")


### print testing results
for i in range(10):
    print(f"x={x_test[i].item() : .2f} "
          f"y_test={y_test[i].item() : .4f} " 
          f"y_predicted={pred_y_test[i].item() : .4f}"
    )


### Badly Fails for all the variations of hidden_dim
#Extrapolation
print("\nExtrapolation : ")
X = torch.linspace(6, 8, 10).reshape(10, 1)
Y = make_labels(X)

with torch.no_grad():
    test_pred = model(X)

print("\nCompare predicted and labels from unseen testing data")
for i in range(10):
    print(f"X={X[i].item() : .2f} "
          f"actual={Y[i].item() : .4f} "
          f"pred={test_pred[i].item() : .4f} "
    )
