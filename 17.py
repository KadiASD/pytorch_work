import torch


def test01():
    x = torch.tensor(10, requires_grad=True, dtype=torch.float32)
    print("x-->", x)
    print(id(x))
    y = 2 * x ** 2
    print("y-->", y)
    print(y.grad_fn)

    print("y.sum()-->", y.sum())
    y.sum().backward()
    print("x的梯度值是：", x.grad)
    x.data = x.data - 0.001 * x.grad
    print(x)
    print(id(x))


test01()
