import numpy as np
import torch
def test():
    data = torch.tensor(np.random.randint(0, 10, [3, 4, 5]))
    print("data-->", data)
    print('data shape-->', data.size)

    new_data = torch.transpose(data, 1, 2)
    print("new_data-->", new_data)
    print("new_data shape-->", new_data.size())

    new_data = torch.transpose(data, 0, 1)
    new_data = torch.transpose(new_data, 1, 2)
    print("new_data-->", new_data)
    print("new_data shape-->", new_data.size())

    new_data = torch.permute(data, [1, 2, 0] )
    print("new_data-->", new_data)
    print("new_data shape-->", new_data.size())

test()
