# From https://github.com/joonleesky/train-procgen-pytorch/blob/master/common/model.py
import torch.nn as nn
import torch
from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaModel(nn.Module):
    def __init__(self, args, action_space):
        super(ImpalaModel, self).__init__()
        self.atoms = args.atoms
        self.action_space = action_space
        self.block1 = ImpalaBlock(in_channels=3, out_channels=16)
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
        self.block3 = ImpalaBlock(in_channels=32, out_channels=32)
        # Value
        self.fc1_v = nn.Linear(in_features=32 * 8 * 8, out_features=256)
        self.fc2_v = nn.Linear(in_features=256, out_features=self.atoms)
        # Advantage
        self.fc1_a = nn.Linear(in_features=32 * 8 * 8, out_features=256)
        self.fc2_a = nn.Linear(in_features=256, out_features=action_space * self.atoms)

    def forward(self, x, log=False):
        # Remove the second dimension of a torch vector if it has 5 dimensions
        if len(x.shape) == 5:
            x = x.squeeze(1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        # Advantage
        x_a = nn.ReLU()(self.fc1_a(x))
        x_a = nn.ReLU()(self.fc2_a(x_a))
        x_a = x_a.view(-1, self.action_space, self.atoms)
        # Value
        x_v = nn.ReLU()(self.fc1_v(x))
        x_v = nn.ReLU()(self.fc2_v(x_v))
        x_v = x_v.view(-1, 1, self.atoms)
        # Combine streams
        q= x_v + x_a - x_a.mean(1, keepdim=True)

        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        return q

    def reset_noise(self):
        pass
