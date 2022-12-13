import torch


class ConvBlock(torch.nn.Module):
    """Unidimensional convolution block."""

    def __init__(self, input_channels, output_channels, window_size):
        """
        Args:
          input_channels  (int): input  depth.
          output_channels (int): output depth.
          window_size     (int): size of window.
        """
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=window_size,
        )
        self.bn = torch.nn.BatchNorm1d(output_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = torch.nn.functional.relu(y)
        return y


class MultiScaleConvBlock(torch.nn.Module):
    """Unidimensional multi-scale convolution block."""

    def __init__(self, input_channels, output_channels, window_sizes):
        """
        Args:
          input_channels  (int): input  depth.
          output_channels (int): output depth.
          window_sizes    (int): sizes of window to be used for multi-scale.
        """
        super(MultiScaleConvBlock, self).__init__()
        self.convs = torch.nn.ModuleList(
            [  # list of the different convolutions
                torch.nn.Conv1d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=w,
                    padding="same",  # to allow concatenation
                )
                for w in window_sizes
            ]
        )
        # this is used to linearly project the concatenated result
        self.merger = torch.nn.Conv1d(
            in_channels=output_channels * len(window_sizes) + input_channels,
            out_channels=output_channels,
            kernel_size=1,
        )
        self.bn = torch.nn.BatchNorm1d(output_channels)

    def forward(self, x):
        # x is concanted with the multi-scale
        # convolution to allow for original information
        # to pass forward. also, to offer a residual
        # connection
        y = torch.cat(
            tensors=[conv(x) for conv in self.convs] + [x],
            dim=1,
        )
        y = self.merger(y)
        y = self.bn(y)
        y = torch.nn.functional.relu(y)
        return y
