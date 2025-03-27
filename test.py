import torch

def test():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    t1 = torch.tensor([1, 2]).to('cuda:0')
    print(t1.device)


if __name__ == '__main__':
    test()