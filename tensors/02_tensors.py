import torch

a = torch.tensor([[1,2,3], [4,5,6]])

b = torch.tensor([[2,5,7], [7,4,1]])

c = torch.tensor([[1,2], [5,3], [4,5]])

print("Sum=", a+b)
print("Diff", a-b)
print("Mul=", a*b)
print("Div=", a/b)
print("Exp=", a**b)

#Dim A(n, m) @ Dim B(m, n)
print("\nMatmul = ", torch.matmul(a,c))

print("Transpose = ", a.T)

print("sum of mat a = ", torch.sum(a))
print("max of mat a = ", torch.max(a))
print("min of mat a = ", torch.min(a))

#torch.mean(a)
#will raise an error because PyTorch computes the mean only for floating-point or complex tensors.
#To compute mean of tensor, it must be converted to either float or int data type
print("\nmean of mat a = ", torch.mean(a.float()))

print("std of mat b = ", torch.std(b.float()))

print("var of mat b = ", torch.var(b.float()))

##
## argmax only accepts atmost 1 arg.
##

#argmax or argmin gives the index of max or min value in tensor
#if 2D tensor, then it flattens into 1D tensor, row-wise flattened
print("argmax of c = ", torch.argmax(c))
print("argmin of c = ", torch.argmin(c))

#we can specify the dimension for finding argmax or argmin also
a = torch.tensor([
    [4, 7, 9], 
    [1, 7, 2]
])
#here dim=0, means collapse the tensor along dim=0 which is row
#so, collapse rows, finally tensor will only have one row
#so, get max value out of each column and make the final row
print("\nargmax a of along dim=0 : ", torch.argmax(a, dim=0))

#similarly, dim=1 means collapse columns.
#find max along each row and build the final column
print("argmin a of along dim=1 : ",torch.argmin(a, dim=1))


#Range starts from 0 and excludes N
print("\narange = ", torch.arange(10))

#lin spacing is inclusive of (a, b) both
print("lin spacing = ", torch.linspace(0, 1, 6))

print("identity mat = ", torch.eye(4))

print("mat full of specific number = ", torch.full((3,4), 7))

#empty means 0
print("empty mat = ", torch.empty(2,3))

#Number of elements in the tensor
print("\nnumel : num el in tensor a = ", a.numel())

a = torch.zeros(4, 5, 6)

print(a.shape)#[4,5,6]
print(a.ndim)#3
print(a.numel())#120