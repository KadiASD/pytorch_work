import numpy as np
import torch
def test():
    data = torch.tensor(np.random.randint(0,10,[1,3,1,5]))
    print("data-->",data)
    print("data_shape:",data.size())

    new_data = data.squeeze()
    print("new_data-->",new_data)
    print("new_data shape:",new_data.size())

    new_data = data.squeeze(2)
    print("new_data-->",new_data)
    print("new_data shape:",new_data.size())

    new_data = data.squeeze(-1)
    print("new_data-->",new_data)
    print("new_data shape:",new_data.size())
    if __name__ == '__main__':
        test()