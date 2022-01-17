import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.fc = nn.Linear(2, 1)


def svr_loss(x, y, e=1):
    loss = max(0, abs((y - x)) - e)
    return loss

if __name__ == '__main__':
    x = 11
    y = 1
    loss = svr_loss(x, y)
    print(loss)