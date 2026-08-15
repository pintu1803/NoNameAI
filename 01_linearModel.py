import torch

X = torch.tensor([[1.],
                  [2.],
                  [3.],
                  [4.],
                  [5.]])

Y = torch.tensor([[3.],
                  [5.],
                  [7.],
                  [9.],
                  [11.]])

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

lr = 0.01

for epoch in range(100):
    print("\nTraining epoch : ", epoch)
    
    w.grad.zero_()
    b.grad.zero_()

    y = X*w + b

    loss = ((Y - y) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        w = w - lr * w.grad
        b = b - lr * b.grad

    print("Updated weights are : w={}, b={}".format(w, b));

