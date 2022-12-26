import torch
import torch.nn.functional as F


class SeBlock(torch.nn.Module):
    """
    1D SeBlock
    """
    def __init__(self, in_channels):
        super(SeBlock, self).__init__()

        self.global_avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.conv1 = torch.nn.Conv1d(in_channels, in_channels // 16, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv1d(in_channels // 16, in_channels, kernel_size=1, stride=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out

class ResidualSEBlock(torch.nn.Module):
    """
    The residual block used by ProtCNN (https://www.biorxiv.org/content/10.1101/626507v3.full).
    
    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first convolution
        dilation: Dilation rate of the first convolution
    """
    
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()   
        
        # Initialize the required layers
        self.skip = torch.nn.Sequential()
            
        self.bn1 = torch.nn.BatchNorm1d(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, bias=False, dilation=dilation, padding=dilation)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels, 
                               kernel_size=3, bias=False, padding=1)
        self.seblock = SeBlock(in_channels=128)
        
    def forward(self, x):
        # Execute the required layers and functions
        activation = F.relu(self.bn1(x))
        x1 = self.conv1(activation)
        x2 = self.conv2(F.relu(self.bn2(x1)))
        x2 = self.seblock(x2)
        
        
        return x2 + self.skip(x)