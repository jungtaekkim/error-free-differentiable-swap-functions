import torch
from torch import nn
import torch.nn.functional as F


class MultiDigitMNISTNet(nn.Module):
    def __init__(self, n_digits=4, num_last=16, num_layers=6):
        super().__init__()

        self.n_digits = n_digits
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.fc1 = nn.Linear(n_digits * 7 * 7 * 64, num_last)

        tel = nn.TransformerEncoderLayer(d_model=num_last, nhead=8)
        self.te = nn.TransformerEncoder(tel, num_layers=num_layers)

        self.fc2 = nn.Linear(num_last, 1)

    def forward(self, x):
        x_shape = x.shape

        if len(x_shape) == 5:
            x = x.reshape(-1, *x_shape[2:])

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.n_digits * 7 * 7 * 64)
        x = self.fc1(x)

        x = self.te(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = x.reshape(*x_shape[:2], 1)
        return x


class SVHNNet(nn.Module):
    def __init__(self, num_last=16, num_layers=6, is_large=False):
        super().__init__()

        self.is_large = is_large

        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)

        if self.is_large:
            self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
            self.conv4 = nn.Conv2d(128, 256, 5, 1, 2)
            self.fc1 = nn.Linear(3*3*256, num_last)
        else:
            self.conv3 = nn.Conv2d(64, 64, 5, 1, 2)
            self.conv4 = nn.Conv2d(64, 128, 5, 1, 2)
            self.fc1 = nn.Linear(3*3*128, num_last)

        tel = nn.TransformerEncoderLayer(d_model=num_last, nhead=8)
        self.te = nn.TransformerEncoder(tel, num_layers=num_layers)

        self.fc2 = nn.Linear(num_last, 1)

    def forward(self, x):
        x_shape = x.shape

        if len(x_shape) == 5:
            x = x.view(-1, *x_shape[-3:])

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        if self.is_large:
            x = x.view(*x_shape[:-3], 3*3*256)
        else:
            x = x.view(*x_shape[:-3], 3*3*128)

        x = self.fc1(x)

        x = self.te(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x


if __name__ == '__main__':
    print(MultiDigitMNISTNet()(torch.zeros(2, 3, 1, 4*28, 28)).shape)
    print(SVHNNet()(torch.zeros(2, 3, 3, 54, 54)).shape)
