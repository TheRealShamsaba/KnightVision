import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Convolutional body (shared backbone)
        self.conv1 = nn.Conv2d(12, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.res1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.res_bn1 = nn.BatchNorm2d(256)
        self.res2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.res_bn2 = nn.BatchNorm2d(256)

        # Policy Head
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096)

        # Value Head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Residual Block
        residual = x
        x = F.relu(self.res_bn1(self.res1(x)))
        x = self.res_bn2(self.res2(x))
        x += residual
        x = F.relu(x)

        # Policy Head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value Head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value