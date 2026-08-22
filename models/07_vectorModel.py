
import torch
import torch.nn as nn

"""
Here we will learn [x, y] -> [x^2 - y^2]
"""

### Function
def make_labels(X):
    Y = torch.tensor( [ [X[i][0]**2 - X[i][1]**2] for i in range(len(X)) ] )
    return Y

### Prepare dataset and shuffle it
# X1 = torch.linspace(0, 5, 1000).reshape(1000, 1)
# X2 = torch.linspace(0, 4, 1000).reshape(1000, 1)

# X1 = torch.rand(1000, 1)*5
# X2 = torch.rand(1000, 1)*5

dataset_size = 1000

X1 = torch.linspace(-3, 3, 1000).reshape(1000, 1)
X2 = torch.linspace(-1, 1, 1000).reshape(1000, 1)

perm = torch.randperm(dataset_size)
X1 = X1[perm]

perm = torch.randperm(dataset_size)
X2 = X2[perm]

X = torch.tensor([[X1[i],X2[i]] for i in range(dataset_size)])
Y = make_labels(X)

perm = torch.randperm(dataset_size)
x = X[perm]
y = Y[perm]


### without shuffling splitting is like extrapolation
### Split dataset into training, validation and testing
train_index = int(dataset_size * .8)
x_train = x[:train_index]
y_train = y[:train_index]

val_index = int(dataset_size * .9)
x_val = x[train_index:val_index]
y_val = y[train_index:val_index]

x_test = x[val_index:]
y_test = y[val_index:]

### Sequential Model
#256 worked fine
# 128 worked better than 256 for testing data
#64 too worked just fine
#32 fine
#16 fine
# 8 shows significantly wrong predictions
"""
Linear layer takes the input as (input_dim, hidden_dim)
or more precisely: (input_dim, output_dim)
if this layer is hidden/intermediate then its output
goes to next layer as input.
"""
input_dim = 2 
hidden_dim = 128
output_dim = 1
model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

print("\nSequential model's hidden linear layer is : ", model[2])
print("Loss criterion is : ", criterion)


### Start training here
epoch = 15000
for epoch in range(epoch):
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
### model gives final output in the shape: (input_size, output_feature)
print("\nTest inference : ")
for i in range(10):
    values = ", ".join(f"{val : .2f}" for val in x_test[i].tolist())
    print(f"x={values} "
          f"y_test={y_test[i][0] : .2f} " 
          f"y_predicted={pred_y_test[i][0] : .2f}"
    )


### Badly Fails for all the variations of hidden_dim
#Extrapolation
print("\nExtrapolation : ")
extrapol_dim = 30
a = torch.linspace(1, 3, extrapol_dim).reshape(extrapol_dim, 1)
b = torch.linspace(0, 2, extrapol_dim).reshape(extrapol_dim, 1)
X = torch.cat((a,b), dim=1)
Y = make_labels(X)

with torch.no_grad():
    test_pred = model(X)

##Nicely formatted print statement
print("\nCompare predicted and labels from unseen testing data")
for i in range(extrapol_dim):
    values = ", ".join(f"{v:.2f}" for v in X[i].tolist())
    print(f"X={values} "
          f"actual={Y[i][0].item() : .2f} "
          f"pred={test_pred[i][0].item() : .2f} "
    )
