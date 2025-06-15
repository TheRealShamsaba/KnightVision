import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Convolutional body (shared backbone)
        self.conv1 = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)

        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512)
            )
            for _ in range(5)
        ])

        # Policy Head
        self.policy_conv = nn.Conv2d(512, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096)

        # Value Head
        self.value_conv = nn.Conv2d(512, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 512)
        self.value_fc2 = nn.Linear(512, 1)

    def forward(self, x):
        with amp.autocast(device_type='cuda', dtype=torch.float16):
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

            policy = policy.to(x.device)
            value = value.to(x.device)

            return policy, value