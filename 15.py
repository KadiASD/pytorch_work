import torch
def test():
    data = torch.tensor([[10, 20, 30],[40, 50, 60]])
    print("data-->", data)
    print('data shape:', data.size())

    new_data = data.view(3, 2)
    print("new_data-->", new_data)
    print('data shape:', new_data.size())
    print("data-->", data.is_contiguous())
    new_data = torch.transpose(data, 0, 1)
    print("new_data-->", new_data)
    print("new_data shape-->", new_data.size())
    print("new_data-->", new_data.is_contiguous())
    print(new_data.contiguous().is_contiguous())
    new_data = new_data.contiguous().view(2, 3)
    print("new_data-->", new_data)
    print("new_data shape-->", new_data.shape)

test()