import torch

"""
Permute:
It is generalisatio of transpose
Transpose just tells which two dimensions need to be swapped
Permute, on the other hand, gives a full list of reordered dimension positions.

----index--0 1 2
x.shape = [2,3,4]

y = x.permute(2,0,1)
means 
New dim-0 ← Old dim-2
New dim-1 ← Old dim-0
New dim-2 ← Old dim-1

y shape = [4, 2, 3]

Permute is not same as reshape.

"""

## 3D tensor
x = torch.arange(24).reshape(2,3,4)
y = x.permute(2,0,1)

print("\n3D tensor, x = \n", x)
print("\nPermuted tensor, y = \n", y)
print("\nNum el are same in x and y ? ", x.numel() == y.numel())

## 4D tensor
x = torch.arange(120).reshape(2,3,4,5)
y = x.permute(3,0,2,1)

print("\nShape of x = ", x.shape)
print("\nShape of y = ", y.shape)

print("\n4D tensor, x = \n", x)
print("\nPermuted tensor, y = \n", y)
print("\nNum el are same in x and y ? ", x.numel() == y.numel())


##################################################################

"""
stride
contiguous

stride() - it tells us how much the pointer moves depending on indices.
it gives a tuple of size same as ndim, each entry tells the jump value in linear memory allocation.
it does not change the memory layout of the tensor, it just changes dim and stride order.

is_contiguous() - it tells if the tensor rows are stored in the same order (as of rows) in physical memory
contiguous() - it allocates new physical memory for original tensor and puts the rows in same order
and thus, the stride tuple also changes, so this method does 2 things.

permute() changes the view of the data.
contiguous() changes the layout of the data.

transpose and permute are fast - because they just change the logical structure of the tensor and strides.
the physical memory layout or data movement does not happen in physical memory.
"""

#Stride
"""
Stride is computed from right -> left for convenience.
For dim=d, the stride is produce of all dimensions remaining on right side.
This formula only holds for contiguous tensors.
After transpose() or permute(), strides are simply rearranged and no longer
follow this product rule.
"""
x = torch.arange(24).reshape(2,3,4)
print("\nShape of x: ", x.shape, "Stride of x:", x.stride())
#x stride = (12, 4, 1)

y = x.permute(1,2,0)
print("\nShape of y: ", y.shape, "Stride of y:", y.stride())
print("is y contigous:", y.is_contiguous())
#False, permute just reorders the dimension axis
#y shape = (3,4,2)
#y stride = (4, 1, 12)


# Copies/rearranges the data into a new contiguous block of memory.
# Assigns new contiguous strides for that layout.
z = x.permute(1,2,0).contiguous()
print("\nShape of z: ", z.shape, "Stride of z:", z.stride())
print("is z contiguous:", z.is_contiguous())
#stride of z = (8, 2, 1)


#offset of x[0][2][1] = 9 => move in contiguous memory by 9 steps
#position of x[1][2][3] = 12*1 + 4*2 + 1*3 = 23 => move to memory address at 23 position from start
#position of y[1][2][3] = 8*1 + 2*2 + 1*3 = 15
print("\nstride of x[0] = ", x[0].stride())

#x[0] has shape (3, 4).
#It is just the first 2D slice of x.
#Since the first dimension is removed, only the remaining strides survive.
print("stride of x[1] = ", x[1].stride())

#Similary, x[1][1] = x[1][0] = x[0][1] = 1
print("\nstride of x[1][1] = ", x[1][1].stride())
print("stride of y[0] = ", y[0].stride())