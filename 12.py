import torch
def test():
    data = torch.tensor([[10,20,30],[40,50,60]])
    print("data-->",data)
    print(data.shape, data.shape[0],data.shape[1])
    print(data.size(),data.size(0),data.size(1))
    new_data = data.reshape(1,6)
    print("new_data-->",new_data)
    print(new_data.shape)
    print(data.reshape(1,-1))
    print(data.reshape(-1,1))
    if __name__ == '__main__':
        test()
