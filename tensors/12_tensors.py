import torch


"""
x.requires_grad_(False) -> freezes the operation tracking on x permanently until unfreezed explicitly.
it turns off requires_grad flag, so, all future operations involing x are not tracked.
this operation acts directly in-place on x and returns x itself.

x.requires_grad_(True) -> unfreezing of operation tracking
now, again, future operations involving x will be tracked.

important: whatever computation graph was created after unfreeze and before freeze will not be affected,
by future freezing or unfreezing
"""

x = torch.tensor([1., 2., 3.])
print("x requires_grad : ", x.requires_grad)

print("\nperform : y = x*3")
y = x * 3
print("y requires_grad : ", y.requires_grad)

print("\nLets set requires_grad True for x")
x.requires_grad_(True)
z = x * 4
print("x requires_grad : ", x.requires_grad)
print("z requires_grad : ", z.requires_grad)

# requires_grad_() method modifies the original tensor in-place
w = x.requires_grad_(False)
print("\nwhat is w = ", w, " w is just another pointer to x itself")

print("operation tracking is freezed for x")
w = x * 4
print("w requires_grad : ", w.requires_grad)
print("w grad_fn : ", w.grad_fn)
print("w value : ", w)
print("w is a leaf : ", w.is_leaf)



### TRICKY TRAP
# x = torch.tensor([1., 2., 3.])
x = torch.tensor(2.0)
x.requires_grad_(True)

z = x ** 2
x.requires_grad_(False)


# claim #1
#since graph exists, it differentiates but does not store grad in x.grad as it's turned off
z.backward()
print("\nx grad after x.requires_grad freezed (but computation graph existed) : ", x.grad) # None
print("torch version : ", torch.__version__)

#explicit differentiation of z wrt x throws error as operation tracking for x is disabled
#reason is that autograd.grad() generally computes and returns derivative, 
#but since x does not have requires_grad = True, it denies computing upfront
try:
    dz_dx = torch.autograd.grad(z, x)[0]
    print("dz_dx = ", dz_dx)
except Exception as e:
    print("Exception e : ", e)



## Code to confirm - claim #1
# x = torch.tensor(2.0)
# x.requires_grad_(True)
# z = x ** 2
# x.requires_grad_(False)

# def hook(grad_inputs, grad_outputs):
#     print("PowBackward0 actually ran")
#     print("  grad_outputs (incoming):", grad_outputs)
#     print("  grad_inputs (computed):", grad_inputs)

# z.grad_fn.register_hook(hook)
# z.backward()
# print("x.grad after backward:", x.grad)