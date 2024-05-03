import torch
from torch import nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    def __init__(self, num_fc_first):
        super().__init__()

        self.num_fc_first = num_fc_first

        self.conv1 = nn.Conv2d(1, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.fc1 = nn.Linear(self.num_fc_first, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x_shape = x.shape

        if len(x_shape) == 5:
            x = x.reshape(-1, *x_shape[-3:])

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(*x_shape[:-3], self.num_fc_first)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class MNISTTrans(nn.Module):
    def __init__(self, num_fc_first, num_last, num_layers):
        super().__init__()

        self.num_fc_first = num_fc_first

        self.conv1 = nn.Conv2d(1, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.fc1 = nn.Linear(self.num_fc_first, num_last)

        tel = nn.TransformerEncoderLayer(d_model=num_last, nhead=8)
        self.te = nn.TransformerEncoder(tel, num_layers=num_layers)

        self.fc2 = nn.Linear(num_last, 1)

    def forward(self, x):
        x_shape = x.shape

        if len(x_shape) == 5:
            x = x.reshape(-1, *x_shape[-3:])

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(*x_shape[:-3], self.num_fc_first)

        x = self.fc1(x)
        x = self.te(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

class CIFAR10Net(nn.Module):
    def __init__(self, num_fc_first):
        super().__init__()

        self.num_fc_first = num_fc_first

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.fc1 = nn.Linear(self.num_fc_first, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x_shape = x.shape

        if len(x_shape) == 5:
            x = x.reshape(-1, *x_shape[-3:])

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(*x_shape[:-3], self.num_fc_first)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class CIFAR10Trans(nn.Module):
    def __init__(self, num_fc_first, num_last, num_layers):
        super().__init__()

        self.num_fc_first = num_fc_first

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.fc1 = nn.Linear(self.num_fc_first, num_last)

        tel = nn.TransformerEncoderLayer(d_model=num_last, nhead=8)
        self.te = nn.TransformerEncoder(tel, num_layers=num_layers)

        self.fc2 = nn.Linear(num_last, 1)

    def forward(self, x):
        x_shape = x.shape

        if len(x_shape) == 5:
            x = x.reshape(-1, *x_shape[-3:])

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(*x_shape[:-3], self.num_fc_first)

        x = self.fc1(x)
        x = self.te(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    print(MNISTNet(4 * 4 * 64)(torch.zeros(2, 4, 1, 14, 14)).shape)
    print(MNISTNet(2 * 2 * 64)(torch.zeros(2, 16, 1, 7, 7)).shape)
    print(MNISTTrans(4 * 4 * 32, 32, 6)(torch.zeros(2, 4, 1, 14, 14)).shape)
    print(MNISTTrans(2 * 2 * 32, 32, 6)(torch.zeros(2, 16, 1, 7, 7)).shape)

    print(CIFAR10Net(4 * 4 * 64)(torch.zeros(2, 4, 3, 16, 16)).shape)
    print(CIFAR10Net(2 * 2 * 64)(torch.zeros(2, 16, 3, 8, 8)).shape)
    print(CIFAR10Trans(4 * 4 * 32, 32, 6)(torch.zeros(2, 4, 3, 16, 16)).shape)
    print(CIFAR10Trans(2 * 2 * 32, 32, 6)(torch.zeros(2, 16, 3, 8, 8)).shape)
