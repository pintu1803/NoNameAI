import torch

"""
Transpose swaps exactly 2 dim
It accepts two args exactly, not more not less
x.transpose(1,2,3) -> fails
x.transpose() -> fails
"""

x = torch.arange(24).reshape(4,2,3)
y = x.transpose(0,1)
z = x.transpose(1,2)
w = x.transpose(0,2)

print("\nShape of x:", x.shape)
print("\nShape of y:", y.shape)
print("\nShape of z:", z.shape)
print("\nShape of w:", w.shape)

#Can't accept more than 2 args
try:
    y = x.transpose(0,1,2)
except Exception as e:
    print("\nException occurred as : ", e)

#Zero args fail

try:
    x = x.reshape(6,4)
    y = x.transpose()
except Exception as e:
    print("\nShape of x:", x.shape)
    print("\nException occurred as : ", e)




    