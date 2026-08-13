import torch


"""
torch.no_grad() temporarily blocks operations tracking,
the scope of the blockage is the code-block with torch.no_grad()
outside this scope, operation tracking is restores and tracks operations thereafter.

Only the operations performed inside the context aren't tracked.
That's why it's called a context manager — the behavior is temporary.
"""
x = torch.tensor(2.0, requires_grad=True)
y = x * 3

print(y.requires_grad) # True
print(y.grad_fn) # MulBackward0

with torch.no_grad():
    z = x * 4

print(z.requires_grad)  # False
print(z.grad_fn)    # None

"""
the tensors created withing the scope behaves like scalar tensors, having:
requires_grad = False
grad_fn = None
is_leaf = True

Any tensors created using such tensor is also of same nature.
Autograd cannot reconstruct history after it has been discarded.
"""
x = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    y = x * 3

z = y * 4

print("\nexample-2")
print(y.requires_grad) # False
print(y.grad_fn) # None
print(y.is_leaf) # True
print(z.requires_grad)
print(z.grad_fn)
print(z.is_leaf)

"""
If any operation/expression involves atleast one tensor which has requires_grad = True
then the operation will be tracked as part of the computation graph of that trackable tensor.
"""

x = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    y = x * 3
    # y is leaf tensor

z = y * x

print("\nexample-3")
print(y.requires_grad) # False
print(z.requires_grad) # True (as x requires tracking)
print(z.grad_fn) # MulBackward0

"""
while computing gradient, for loss (or any other variable name)
d(loss)/d(leaf tensor) -> this is computed for all leaf tensors with requires_grad = True

z = y * x
y and x are leaf tensors but only x has requires_grad = True
dz/dx = y
"""
x = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    y = x * 3

z = y * x
z.backward()

print("\nexample-4")
print(x.grad) # 6.0

"""
detach() creates a new tensor after detaching it from computation graph
torch.no_grad() temporarily blocks graph tracking within its scope
so, effectively both create the new tensors with grad_fn = None, 
thus creating leaf tensors.

Once backward() is called the computation graph is freed, if not otherwise retained explicitly.
"""

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)

try: 
    y.backward()
    print(x.grad)
except Exception as e: 
    print("\nException occurred while performing backward again on y:", e)

## Retain the graph for doing backward again
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)
print("\nFirst time loss:", x.grad)
y.backward()
print("Second time loss accumulated:", x.grad)

"""
requires_grad is an attribute.
to toggle it, use method: requires_grad_()
"""

x = torch.tensor(1.0, requires_grad=True)
print("\nrequires_grad of x : ", x.requires_grad)
x.requires_grad_(False)
print("requires_grad of x : ", x.requires_grad)
x.requires_grad_(True)
print("requires_grad of x : ", x.requires_grad)


"""
torch.inference_mode():
the tensors created under this scope are done for autograd operations.
they are:
leaf tensors
grad_fn = None
requires_grad = False
not allowed to be used in any other expressions which requires grad tracking*
only allowed for clone

y * <plain number> outside the block → fine, ordinary tensor op, no graph needed
y * <tensor with requires_grad=True> outside the block → error
y.requires_grad_(True) → error, you're trying to force y into being trackable
"""

x = torch.tensor(2.0, requires_grad=True)
with torch.inference_mode():
    y = x * 3

# y * scalar -> allowed
z = y * 3
print("\nz created using inference tensor:", z)

# y is not allowed to be used in expression
try:
    z = y * x
except Exception as e:
    print("\ninference tensor used with trackable tensor in an expression : ", e)

# inference tensor forced to track grad
try: 
    y.requires_grad_(True)
except Exception as e:
    print("\ninference tensor turned on for tracking grad:", e)
