import torch


def test():
    data = torch.randint(0, 10, [2, 3], dtype = torch.float64)
    print(data)
    print('-'*50)
    print(data.mean())
    print(data.mean(dim=0))
    print(data.mean(dim=1))
    print('-'*50)
    
    print(data.sum())
    print(data.sum(dim=0))
    print(data.sum(dim=1))
    print('-'*50)


    print(data.pow(exponent=2))
    print('-'*50)


    print(data.sqrt())
    print('-'*50)