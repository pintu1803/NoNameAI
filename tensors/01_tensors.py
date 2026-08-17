import numpy as np
import torch as t


"""
Summary
Property	Method/Attribute	Returns	Example             Output
=================================================================================
shape	    Attribute	        Shape of the tensor	        torch.Size([2, 3])
size()	    Method	            Same as shape	            torch.Size([2, 3])
size(dim)	Method	            Size of one dimension	    2
dim()	    Method	            Number of dimensions	    2
ndim	    Attribute	        Same as dim()	            2
type()	    Method	            Tensor type	                torch.LongTensor
dtype	    Attribute	        Data type	                torch.int64

t.ones(m,n)
t.rand(m,n)
"""

# print(np.__version__)
a = np.array([1,2,3])
print(a)

a = t.tensor(5)
print(a)
print("type of a=", a.type())

a = t.tensor([1,2,3,4])
print("\ntensor a= ",a)
print("type of a=", a.dtype)
print("shape of a=", a.size())

a = t.tensor([[1,2,3], [2,3,4]])
print("\ntensor a= ",a)
print("type of a=", a.dtype)
print("shape of a=", a.shape)

print("\ndim of a=", a.dim())
print("size of a=", a.size())
print("ndim of a=", a.ndim)

a = t.ones(5,5)
print("\na (5,5) filled with 1s=", a)
print("shape of a=", a.size())
print("type of ones tensor=", a.dtype)

# Creates random numbers uniformly distributed between 0 and 1
b = t.rand(3,4)
print("\nrandom b(3,4)=", b)
print("shape of b=", b.size())
print("type of random tensor=", b.dtype)

# Creates random numbers from a standard normal distribution (mean=0, std=1)
b = t.randn(3,4)
print("\nNormal distribution random b(3,4)=", b)
print("shape of b=", b.size())
print("type of random tensor=", b.dtype)

c = t.zeros(2,3)
print("\nc (2,3) zero tensor = ", c)
print("shape of c= ", c.shape)
print("type of c=", c.dtype)

