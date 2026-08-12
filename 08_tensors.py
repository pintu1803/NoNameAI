import torch

"""
requires_grad meaning:
This is a learnable parameter.
Track every differentiable operation performed on it.

Default value for requires_grad is false
Mention it or don't: requires_grad=False; it's the same thing.

When requires_grad=True is passed explicitly, torch stores a computation graph for given tensor.
Starts tracking operations and builds the computation graph.

Leaf tensors have grad function as None. But, they store gradients by default.

Intermediary tensors are those which are created by expressions/equations involving leaf tensors.
Intermediary tensors usually have grad_fn, but do not store gradients in .grad unless you explicitly call retain_grad().

loss.backward() computes the derivative of loss wrt every leaf tensor and updates the grad value of those tensors.
Traverses the computation graph backwards using the chain rule and computes gradients.

During backward(), torch walks this graph backwards using the chain rule.
This is why it's called backpropagation.

for tensor x, PyTorch does not replace the old gradient, when computed another gradient from another equation.
it accumulates gradients automatically, if we want to switch it off, explicit intervention needed.
One way of turning of on tensor itself:
x.grad.zero_()

Second way is to cleanup old grad using optimizer.
optimizer.zero_grad()

"""

#Explicit passing of requires_grad=True
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
y.backward()
print("\nDerivative of y (=x^3) wrt x(=2.0) = ", x.grad)

## Grad function exists for all tensors
## grad_fn attribute exists on every tensor, but its value is None for leaf tensors
print("Grad function of x = ", x.grad_fn)
print("Grad function of y (intermediary tensor) = ", y.grad_fn)


## You explicitly asked torch not to track computation graph for x.
try:
    x = torch.tensor(2.0, requires_grad=False)
    y = x ** 3
    y.backward()
    print("Derivative of y (=x^3) wrt x(=2.0) = ", x.grad)
except Exception as e:
    print("\nBad exception occurred : ", e)


## Only float tensors can be used for derivative computation
## this fails while creating the tensor itself.
## tensor can't be created with integer values.
## integer tensors cannot have requires_grad=True (only float/complex can).
try:
    x = torch.tensor(2, requires_grad=True)
except Exception as e:
    print("\nSerious exception occurred : ", e)


##Gradient accumulation
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
y.backward()
print("\nderivative of y (=x^3) wrt x(=2.0) = ", x.grad) # 12  (dy/dx = 3x^2 = 12)

z = 3*x
z.backward()
print("derivative of z (=3x) wrt x(=2.0) = ", x.grad)  # 15, NOT 3


## Turn off gradient accumuation for a given tensor.
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print("\nderivative of y (=x^2) wrt x(=3.0) = ", x.grad)
x.grad.zero_()

z = 3*x
z.backward()
print("derivative of z (=3x) wrt x(=3.0) = ", x.grad)


## Grad of complex equations
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
z = 2*y + 3
w = y ** 2 + z ** 2
w.backward()
print("\nDerivative of w wrt. x(3.0) = ", x.grad)

