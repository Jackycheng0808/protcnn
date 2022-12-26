import torch
import torch.nn.functional as F

from .layers.residual_block import ResidualBlock
from .layers.residual_se_block import ResidualSEBlock


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ResNet(torch.nn.Module):
    """
    The residual block used by ProtCNN (https://www.biorxiv.org/content/10.1101/626507v3.full).

    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first convolution
        dilation: Dilation rate of the first convolution
    """

    def __init__(self, num_classes=17930, ratio=1):
        super().__init__()

        last_channel = round(7680 * ratio)

        # Initialize the required layers
        self.conv1 = torch.nn.Conv1d(22, 128, kernel_size=1, padding=0, bias=False)

        self.res1 = ResidualBlock(128, 128, dilation=2)
        self.res2 = ResidualBlock(128, 128, dilation=3)

        self.maxPool = torch.nn.MaxPool1d(3, stride=2, padding=1)
        self.linear = torch.nn.Linear(last_channel, num_classes)

    def forward(self, x):
        # Execute the required layers and functions
        out = self.conv1(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.maxPool(out)
        out = Lambda(lambda x: x.flatten(start_dim=1))(out)
        out = self.linear(out)

        return out


class ResSENet(torch.nn.Module):
    """
    The residual block used by ProtCNN (https://www.biorxiv.org/content/10.1101/626507v3.full).

    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first convolution
        dilation: Dilation rate of the first convolution
    """

    def __init__(self, num_classes=17930, ratio=1):
        super().__init__()

        last_channel = round(7680 * ratio)

        # Initialize the required layers
        self.conv1 = torch.nn.Conv1d(22, 128, kernel_size=1, padding=0, bias=False)

        self.res1 = ResidualSEBlock(128, 128, dilation=2)
        self.res2 = ResidualSEBlock(128, 128, dilation=3)

        self.maxPool = torch.nn.MaxPool1d(3, stride=2, padding=1)
        self.linear = torch.nn.Linear(last_channel, num_classes)

    def forward(self, x):
        # Execute the required layers and functions
        out = self.conv1(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.maxPool(out)
        out = Lambda(lambda x: x.flatten(start_dim=1))(out)
        out = self.linear(out)

        return out
