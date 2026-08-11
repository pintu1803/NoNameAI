import torch

"""
| Operation    | Allowed? |
| ------------ | -------- |
| `x[-1]`      | ✅ Yes    |
| `x[:, -2]`   | ✅ Yes    |
| `x[1:-1]`    | ✅ Yes    |
| `x[::-1]`    | ❌ No     |
| `x[:, ::-1]` | ❌ No     |
"""


x = torch.arange(20).reshape(4,5)

print("\nx = \n", x)
print("\nx[1] = \n", x[1])

print("\nx[1:3]=\n", x[1:3])

print("\nx[:, 1::2]=\n", x[:, 1::2])

print("\nx[:, 1::2]=\n", x[:, 1::2])

## -1 index is allowed for rows or columes in torch
## but not allowed for step - x[::-1]
print("\nx[:, -1]=\n", x[:, -1])

print("\nx[:2, :3]=\n", x[:2, :3])

#Torch does not support negative-step slicing. Instead, you'd use:
torch.flip(x, dims=[0])
#to reverse the rows.

#[start : end : step] - for both rows and columns
print("\nx[1:4:1, 2:3:2] = \n", x[1:4:1, 2:3:2])


#Row - 1
print(x[1])

#interesting: The output is actually single row
#because selecting a single column index removes that dimension.
print("\nx[:, 2] = \n", x[:, 2])

#Instead use this:
print("\nSingle colume, x[:, 2:3] = \n", x[:, 2:3])