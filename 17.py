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


def test02():
    x = torch.tensor([10, 20], requires_grad=True, dtype=torch.float32)
    print("x-->", x)
    y = 2 * x ** 2
    print("y-->", y)
    y.sum().backward()
    print("x的梯度值是：", x.grad)


def test03():
    x1 = torch.tensor(10, requires_grad=True, dtype=torch.float64)
    x2 = torch.tensor(29, requires_grad=True, dtype=torch.float64)
    y = x1 ** 2 + x2 ** 2 + x1 * x2
    y = y.sum()
    y.backward()
    print("x1.grad-->", x1.grad)
    print("x2.grad-->", x2.grad)


def test04():
    x1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float64)
    x2 = torch.tensor([30, 40], requires_grad=True, dtype=torch.float64)
    y = x1 ** 2 + x2 ** 2 + x1 * x2
    print("y-->", y)
    y = y.sum()
    print("y.sum()-->", y.sum())
    y.backward()
    print("x1.grad-->", x1.grad)
    print("x2.grad-->", x2.grad)


'''
计算过程
'''


def test05():
    x = torch.ones(2, 2, requires_grad=True)
    print("x-->", x)
    y = x + 2
    print("y-->", y)
    z = y * y * 3
    print("z-->", z, z.shape)
    out = z.mean()
    print("out-->", out, out.shape)
    out.backward()
    print("x.grad-->", x.grad)


test01()
test02()
test03()
test04()
test05()
