import torch

"""
torch.autograd.grad(y, x)
1. traverses the graph backward
2. computes the derivative
3. returns it
4. destroys the graph

loss.backward()
1. traverses the graph backward
2. computes the derivative
3. stores the gradient into leaf_tensor.grad
4. destroys the graph

Both backward() and autograd.grad() free the graph by default. retain_graph=True prevents that.
"""

x = torch.tensor(2.0, requires_grad=True)

y = x ** 3
z = x ** 2
#two branches of graph exist

z.backward()
print(x.grad)

grad = torch.autograd.grad(y, x)
print(grad)
print(x.grad)

# Graph is freed
try:
    print(y.backward())
except Exception as e:
    print("\nException while computing derivate second time for y: ", e)

# graph is freed
try: 
    print(torch.autograd.grad(z, x))
except Exception as e:
    print("\nException while computing derivative second time for z: ", e)


"""
We know after derivative computation, the graph is freed, but what if we want to compute second derivative.
Obviously we need graph of the first derivate for that.
so, create_graph = True 
Build a new graph while computing this derivative, so I can differentiate the derivative.

retain_graph=True
    ↓
Keep ORIGINAL graph alive
    ↓
Can differentiate y again


create_graph=True
    ↓
Build a NEW graph for the gradient
    ↓
Can differentiate dy_dx
"""
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
dy_dx = torch.autograd.grad(y, x, retain_graph=True)[0] # Result is tuple of one item.
## original Graph is retained, so we can do derivativation again in future

print("\nFirst derivate of y wrt x : ", dy_dx)

## the derivative is computed and returned to dy_dx variable
print("requires_grad without create_graph=True: ", dy_dx.requires_grad)
print("grad_fn without create_graph=True: ", dy_dx.grad_fn)

## dy_dx has no graph of its own
try:
    dy2_dx2 = torch.autograd.grad(dy_dx, x)[0]
except Exception as e:
    print("\ndy_dx has no graph : ", e)

## Remeber: we retained original Graph, so we can differentiate y again wrt x
dy__dx = torch.autograd.grad(y, x, create_graph=True)[0]

print("\ndy__dx computed again from original graph : ", dy__dx)

## we also created graph of this dy__dx derivative itself, for second derivative purpose
print("\nrequires_grad with create_graph=True", dy__dx.requires_grad)
print("grad_fn with create_graph=True", dy__dx.grad_fn)

dy2_dx2 = torch.autograd.grad(dy__dx, x)[0]
print("\nSecond derivate : ", dy2_dx2)

"""
retain_graph=True
        │
        ▼
KEEP existing graph
        │
        └──→ reuse y's graph


create_graph=True
        │
        ▼
CREATE graph for gradient
        │
        └──→ differentiate dy/dx
"""