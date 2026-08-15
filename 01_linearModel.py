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

    w.grad = None
    b.grad = None

    y = X*w + b

    loss = ((Y - y) ** 2).mean()
    print("Training epoch: {}, loss: {} ".format(epoch, loss))

    loss.backward()

    # w = w - lr * w.grad -> this creates new tensor after doing element-wise subtraction.
    # we don't want that, goal is to reduce the original tensor values.
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    print("Updated weights are : w={}, b={}".format(w, b))