import torch

def test():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())


if __name__ == '__main__':
    test()