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

        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256)
            )
            for _ in range(3)
        ])

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

        # Residual Blocks
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x += residual
            x = F.relu(x)

        # Policy Head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = torch.flatten(policy, 1)
        policy = self.policy_fc(policy)

        # Value Head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        print(f"[DEBUG] Input shape: {x.shape}")
        print(f"[DEBUG] Policy output shape: {policy.shape}")
        print(f"[DEBUG] Value output shape: {value.shape}")

        return policy, value