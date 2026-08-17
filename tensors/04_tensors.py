import torch

"""
For element-wise arithmetic operations, 
the exact shape and count of both tensors must be same. 
That is what is achieved virtually by broadcasting. 
If some dimensions are not matching but can be expanded then that is done so.
"""
#Brodcasting
#For performing element wise arithmetic operations
#both tensors must have same dim
#but to avoid repeated content and unnecessary memory usage
#we use broadcasting

#1#
a = torch.tensor([
    [1, 2, 3],
    [5, 2, 4]
])

b = torch.tensor([10, 20, 30])

print("a+b=\n", a+b)
print("a-b=\n", a-b)
print("a*b=\n", a*b)

#2#
a = torch.tensor([
    [
        [1, 2, 3],
        [5, 2, 4]
    ], 
    [
        [3, 7, 2],
        [1, 3, 2]
    ]
])

# a = [2, 2, 3]
# b = [3,] => [1, 1, 3]

b = torch.tensor([10, 20, 30])

print("a+b=\n", a+b)


#3#
a = torch.tensor([
    [
        [1, 2, 3],
        [5, 2, 4]
    ], 
    [
        [3, 7, 2],
        [1, 3, 2]
    ]
])

# a = [2, 2, 3]
# b = [1, 1] => [1, 1, 1] leading 1
# c = [1, 1, 1]

b = torch.tensor([[10]])
c = torch.tensor([[[10]]])

print("a+b=\n", a+b)
print("a+c=\n", a+c)


#4#
a = torch.tensor([
    [
        [1, 2, 3]
    ], 
    [
        [3, 7, 2]
    ]
])

# a = [2, 1, 3]
# c = [4, 3] => [1, 4, 3]

c = torch.tensor([
    [1, 5, 9],
    [2, 3, 1], 
    [8, 4, 1], 
    [-1, 3, 0]
])

print("a+c=\n", a+c)

##Broadcasting conclusion:
"""
a+b check dim from right to left.
if a and b don't have same number of dimensions,
then shorter tensor is considered to have leading 1s as missed dim
every dim should be matched
match means - either one dim is 1, or both are equal

some dim of a get broadcasted to b, and some from b get broadcasted
it's not that always smaller tensor get broadcasted
"""

####
#Reshape
####

x = torch.arange(12)
print("shape of x:", x.shape)
print("x=\n", x)

y = x.reshape(2, 6)
print("y shape=", y.shape)
print("y=\n", y)

z = x.reshape(3, -1)
print("z shape=", z.shape)
print("z=\n", z)

try:
    er = x.reshape(5, 3)
except Exception as e:
    print("Error=", e)

#original tensor is not affected
print("orginal tensor x = \n", x)

print("3 dim reshaping=", x.reshape(2,3,2).shape)
print("4 dim reshaping=", x.reshape(2,1,3,2).shape)
print(x.reshape(2,1,3,2))

print("\nSize of x and y are same ? ", x.numel() == y.numel())


#####
#Flatten
#####

x = torch.arange(60).reshape(2,3,2,5)
y = x.flatten()

print("\nFlatten code: y=", y)

print("\nflatten dim=0 to last : \n", x.flatten(0))
print("\nflatten dim=0 to 2 : \n", x.flatten(0, 2))
print("\nflatten dim=0 to 2 : \n", x.flatten(start_dim=0, end_dim=2))

print("\nflatten dim=1 to 3 : \n", x.flatten(start_dim=1, end_dim=3))
print("\nflatten dim=1 to last : \n", x.flatten(start_dim=1, end_dim=-1))
print("\nflatten dim=1 to last : \n", x.flatten(1, -1))

try:
    print("err:", x.flatten(2,3,4))
except Exception as e:
    print("\nException occurred during flattening - ", e)

"""
flatten() accepts a range of dim that needs to be flattened
keyword args are start_dim and end_dim
it only accepts 2 args, 3 args will give error.
if 1 arg is passed, 2nd will be taken -1 as default
-1 means last dim
dimension are 0-indexed
(2,3,4,5) -> flatten(0,1) -> (6,4,5)
"""

"""
reshape()  -> specify the exact new shape.

flatten() -> merge a consecutive range of dimensions.

numel() -> total number of elements, which never changes after reshape/flatten.
"""