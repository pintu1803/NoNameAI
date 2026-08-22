import torch as t

#random int values generates
#duplicates allowed
#gives shape (5,2)
#parameters taken: (low, high, shape) -> high is excluded
x = t.randint(10, 20, (5,2))
print(x)

#does not accept dtype = t.float32
#convert into float later
x1 = x.float()
print(x1)

# RuntimeError: a Tensor with 10 elements cannot be converted to Scalar
# print(x.item()) 

print(x1[1][1].item())
print(x1[1,1].item())
print(x1.tolist())

#concatenate two datasets along one axis
#all other dimensions must be same
d1 = t.tensor([[1],
               [2],
               [3],
               [4]])
d2 = t.tensor([[5],
               [7],
               [1],
               [9]])

d3 = t.cat((d1, d2, d1), dim=1)
d4 = t.cat((d1, d2), dim=1)

print(d1)
print(d2)
print(d3)
print(d4)

## item() only works on scalar
## either use pure scalar tensor
## or access index to get scalar
# print(d3.item())
# print(d3[0].item())

#list can be made from n-dimensional tensor
#only last dim is flattened, (leftmost dim is omitted, others remain)
print(d3.tolist())
print(d3[0].tolist())