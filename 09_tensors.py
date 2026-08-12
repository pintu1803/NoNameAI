import torch

"""
A Leaf tensor is the one which is created by user manually -> informal definition
A leaf tensor is the one which has grad_fn = None -> formal definition
is_leaf is an attribute of tensor and tells True or False.

A leaf tensor is leaf irrespective of requires_grad value.
By default settings, retain_grad is False for all type of tensors.
but for leaf nodes, requires_grad=True makes retain_grad=True automatically, 
but for non-leaf we need to mention it explicitly

We explicitly tell torch for which tensor we want to store grad.

requires_grad=True :
It tells torch to build and store computation graph of this leaf node, and
store grad_fn as part of computation graph construction, grad_fn is stored for all nodes.
Finally, store grad value of leaf node.

(important):
we pass requires_grad=True in leaf node creation, and for all non-leaf nodes also this becomes 
true. it tells torch which node needs to be kept in computation graph.

Non-leaf: (important)
A tensor is not a leaf only if it was created by an operation that Autograd is tracking.
If Autograd isn't tracking (because none of the inputs require gradients), 
there is no computation graph, and the resulting tensor is considered a leaf.

Bottom line : non-leaf node is part of some computation graph
"""

##leaf and non-leaf node
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0)
z1 = x + 3
z2 = y + 4
z3 = x + y

print("\nIs tensor-x a leaf ? ", x.is_leaf) # True
print("Is tensor-y a leaf ? ", y.is_leaf) # True
print("Is tensor-z1 a leaf ? ", z1.is_leaf) # False
print("Is tensor-z2 a leaf ? ", z2.is_leaf) # True
print("Is tensor-z3 a leaf ? ", z3.is_leaf) # False

## z1 and z3 are non-leaf as they are part of computation graph of x,
#  or they invole atleast one tensor for which autograd is tracked.
## z2 is a leaf, because it is not part of any compution graph.


"""
grad:
For non-leaf nodes, torch computes grad internally but after using it, it discards it to save memory.
If we need the grad to be saved, we explicitly need to tell that after creating (and before backward()) the node.

If we try to access grad value of non-leaf node, whose grad we are not retaining explicitly
pytorch gives us a elaborated warning.
"""

x = torch.tensor(2.0, requires_grad=True)

y = x * 3
z = y ** 2

print("\nBefore backward: grad of x={}, z={}".format(x.grad, z.grad))
z.retain_grad()

z.backward()

print("\nAfter backward: grad of x={}, z={}".format(x.grad, z.grad))


## Warning for accessing grad of non=leaf
## no matter if you try access this after or before backward()
## because .grad is an attribute.
print("\n\n")
print("\nWarning for accessing grad of non-leaf node y :", y.grad)


"""
loss.backward()

Every tensor stores:
"How much would the final loss change if this tensor changed a tiny bit?"

∂(loss)/∂(that tensor)

x.grad stores: ∂z/∂x

y.grad stores: ∂z/∂y

In general, for any tensor T,
T.grad= ∂Loss/∂T

This is probably the most important formula in Autograd.

x.grad = dLoss/dx
y.grad = dLoss/dy
weight.grad = dLoss/d(weight)
bias.grad = dLoss/d(bias)

Notice the numerator is always the final loss.
"""

x = torch.tensor(2.0, requires_grad=True)

y = x * 3
y.retain_grad()

z = y ** 2
z.retain_grad()

loss = z + 10

loss.backward()

print("\nloss = z + 10, z.grad=", z.grad)
print("z = y^2, at y={} y.grad = {}".format(x*3, y.grad))
print("y = 3*x, at x={} x.grad = {} ".format(x, x.grad))
print("\ngrad of z={}, y={}, x={}".format(z.grad, y.grad, x.grad))

"""
Backward Pass
-------------
loss.grad = 1

↓

z.grad = loss.grad x dloss/dz

↓

y.grad = z.grad x dz/dy

↓

x.grad = y.grad x dy/dx
"""

"""
Tensor attributes:

Non-leaf
requires_grad=True
grad_fn=MulBackward0
grad=None
"""

x = torch.tensor(2.0, requires_grad=True)
y = x * 3

print()
print(x.is_leaf)
print(y.is_leaf)

print(x.requires_grad) # True
print(y.requires_grad) # True

print(x.grad_fn) # None
print(y.grad_fn)

print(x.grad) # None
print(y.grad) # Warning

"""
requires_grad=True → "Track this tensor in the computation graph."
retain_grad() → "Even though this is an intermediate tensor, also save its gradient after backpropagation."

detach() means: 
"Create another tensor that has the same data, but is completely disconnected from the computation graph."

detach() disables gradient tracking.

shares the same underlying data
removes gradient history

clone() means:
z = y.clone()
copies the data
keeps gradient history
"""

x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y.detach()

print(y) # tensor(6., grad_fn=<MulBackward0>)
print(z) # tensor(6.)

print(y.requires_grad) # True
print(z.requires_grad) # False

print(y.grad_fn) # <MulBackward0>
print(z.grad_fn) # None

y.backward()
print("\nGrad of x=", x.grad)

z = z.requires_grad_()
z1 = z ** 2
z2 = z1 * 3

print(z.requires_grad) # True
print(z1.requires_grad) # True

print(z1.grad_fn) # <PowBackward0>
print(z2.grad_fn) # <MulBackward0>

z2.backward()

print("\nGrad of z=", z.grad)


"""
z = y.detach().requires_grad_()
creates a graph detached tensor and marks it true for tracking its own computation graph

It doesn't reconnect the graph.
It simply says:
"From now on, treat z as a new leaf tensor that requires gradients."
"""

x = torch.tensor(2.0, requires_grad=True)

y = x * 3

z = y.detach().requires_grad_()

loss = z ** 2

loss.backward()

print(x.grad) #None
print(z.grad) #12.0

y.backward()
print(x.grad) #3


"""
| Property                 | `clone()`        | `detach()` |
| ------------------------ | ---------------- | ---------- |
| Copies data?             | ✅ Yes            | ❌ No       |
| Shares memory?           | ❌ No             | ✅ Yes      |
| Keeps computation graph? | ✅ Yes            | ❌ No       |
| `requires_grad`          | Same as original | `False`    |
| `grad_fn`                | `CloneBackward0` | `None`     |

"""

x = torch.tensor([1., 2., 3.], requires_grad=True)

y = x.detach()

y[0] = 100

print(x) # [100.,2.,3.]
print(y) # [100.,2.,3.]


"""
detach()
Cuts the graph. -> No gradients.

clone()
Copies the memory. -> Independent data.

So after
y = x.detach().clone()

no computation graph
independent memory
modifying y won't affect x
gradients won't flow back to x
"""

x = torch.tensor([1.,2.,3.], requires_grad=True)
y = x.clone()

y[0] = 100

print(x) # [1.,2.,3.]
print(y) # [100.,2.,3.]


"""
grad_fn = None -> tensor is leaf tensor
clone -> copies tensor into new memory + preserves the computation graph state
new memory
requires_grad = True
grad_fn = cloneBackward
grad = None
is_leaf = False

detach -> points to the memory of its parent tensor + disconnects from computation graph
no new memory
requires_grad = False
grad_fn = None
grad = Warning
is_leaf = True

"""
x = torch.tensor(2.0, requires_grad=True)

y = x.clone()

z = y.detach()

print(x.requires_grad) # True
print(y.requires_grad) # True
print(z.requires_grad) # False

print(x.is_leaf) # True
print(y.is_leaf) # False
print(z.is_leaf) # True

print(y.grad_fn) # cloneBackward0
print(z.grad_fn) # None

print(x.grad) # None
print(y.grad) # Warning as y is non-leaf
print(z.grad) # None


### 3 opeartions chained in sequence
x = torch.tensor(2.0, requires_grad=True)

y = x.clone().detach().requires_grad_()

print(y.is_leaf)         # True
print(y.grad_fn)         # None
print(y.requires_grad)   # True


"""
y = x.clone().detach()
y = x.detach().clone()

Both produce the same end result, 
a leaf tensor,
has no computation history,
has its own memory, and
has requires_grad=False.
"""

x = torch.tensor(2.0, requires_grad=True)

y = x.clone().detach()
z = x.detach().clone()

print("\nLeaf: y={}, z={}".format(y.is_leaf, z.is_leaf))
print("\nrequires_grad: y={}, z={}".format(y.requires_grad, z.requires_grad))
print("\ngrad: y={}, z={}".format(y.grad, z.grad))
print("\ngrad_fn: y={}, z={}".format(y.grad_fn, z.grad_fn))