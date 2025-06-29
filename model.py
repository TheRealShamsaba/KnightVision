import torch
import torch.nn as nn

# Enable cuDNN autotuner for better performance on fixed-size inputs
torch.backends.cudnn.benchmark = True

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ChessNet(nn.Module):
    def __init__(self, verbose: bool = False):
        super(ChessNet, self).__init__()
        self.verbose = verbose
        self.conv1 = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)

        self.res_blocks = nn.ModuleList([ResidualBlock(512) for _ in range(5)])

        self.policy_conv = nn.Conv2d(512, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096)

        self.value_conv = nn.Conv2d(512, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 512)
        self.value_fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # ensure input is on the same device and dtype as the model parameters
        param = next(self.parameters())
        x = x.to(device=param.device, dtype=param.dtype)
        if self.verbose:
            print("📥 Forward input shape:", x.shape)
        # Removed autocast to ensure consistent dtype across input and model parameters
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))

        for block in self.res_blocks:
            x = block(x)

        policy = torch.relu(self.policy_bn(self.policy_conv(x)))
        policy = torch.flatten(policy, 1)
        policy = self.policy_fc(policy)
        if self.verbose:
            print("📤 Policy output shape:", policy.shape)

        value = torch.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = torch.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        if self.verbose:
            print("📤 Value output shape:", value.shape)

        return policy, value
