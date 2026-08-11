import torch

"""
Unsqueeze(dim=d) -> add extra dim of size 1 at dimension-index=d
Squeeze() -> remove all dimensions of size 1
Squeeze(dim=d) -> remove dimension at index-d only if its size is 1

Both accept negative indexing for dim value
-1 means last index (last dimension)

Shape       Meaning
()          -> scalar
(5,)        -> Vector
(3,4)       -> Matrix
(3,2,5)     -> 3D tensor

# unsqueeze()
# Insert a 1 into the shape tuple.

# squeeze()
# Remove a 1 from the shape tuple.
"""

x = torch.arange(35).reshape(5,7)
y = x.unsqueeze(dim=2)
print("\nShape of x:", x.shape, "\nShape of y:", y.shape)
# output = x: [5,7], y:[5,7,1]

x = torch.arange(60).reshape(3,2,5,-1)
y = x.unsqueeze(dim=-2)
print("\nShape of x:", x.shape, "\nShape of y:", y.shape)
# output = x: [3,2,5,2], y:[3,2,5,1,2]

#Can't go beyond valid positions.
#For ndim=4, we have 5 possible insertion positions;
# 0, 1, 2, 3, 4
# -5, -4, -3, -2, -1
try:
    x = torch.arange(60).reshape(3,2,5,-1)
    y = x.unsqueeze(dim=5)
except Exception as e:
    print("\nException occurred : ", e)


"""
Squeeze() -> if not dim passed, remove all dim of size 1
Squeeze(dim=d) -> remove dim=d if its size is 1, else do nothing

For tensor T, if the ndim = n, 
then squeeze(dim=d), d can take up values 0, 1, .., n-1
only valid dimensions it can take
"""
######

x = torch.arange(60).reshape(2,3,1,2,1,5,1)
y = x.squeeze()
z = x.squeeze(dim=2)
w = x.squeeze(dim=4)
v = x.squeeze(dim=6)

print("\nShape of x:", x.shape)
print("\nShape of y:", y.shape)
print("\nShape of z:", z.shape)
print("\nShape of w:", w.shape)
print("\nShape of v:", v.shape)

# Can't squeeze the invalid dimension
try:
    y = x.squeeze(dim=8)
except Exception as e:
    print("\nException occurred : ", e)

# No error if specified dimension is not 1
y = x.squeeze(dim=0)
z = x.squeeze(dim=1)
print("\nShape of x:", x.shape)
print("\nShape of y:", y.shape)
print("\nShape of z:", z.shape)

# Very special case, all dim are 1
x = torch.tensor([[[[[1]]]]])
y = x.squeeze()
print("\nShape of x:", x.shape)
print("\nShape of y:", y.shape)
