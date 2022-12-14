import torch
from .layers.multiscale_block import ConvBlock, MultiScaleConvBlock


class MiniMSCNN(torch.nn.Module):
    def __init__(self, voc_size=22, num_classes=17930):
        """
        Args:
          voc_size  (int): size of amino-acids vocabulary.
                           i.e, size of one-hot vectors.
                           i.e, depth of input features.
          num_classes (int): number of output classes.
        """
        super(MiniMSCNN, self).__init__()
        self.embedding_layer = ConvBlock(voc_size, 16, 1)
        self.main_network = torch.nn.Sequential(
            MultiScaleConvBlock(16, 32, [2, 3, 4, 5]),
            torch.nn.MaxPool1d(4, 4),
            MultiScaleConvBlock(32, 64, [2, 3, 4, 5]),
            torch.nn.MaxPool1d(4, 4),
            MultiScaleConvBlock(64, 128, [2, 3, 4, 5]),
            torch.nn.MaxPool1d(4, 4),
            MultiScaleConvBlock(128, 256, [2, 3, 4, 5]),
        )
        self.linear = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Args:
          x  (torch.tensor): of shape (N, V, L).
                            N: batch-size.
                            V: amino-acids vocabulary size.
                            L: sequence length
        """
        y = self.embedding_layer(x)
        y = self.main_network(y)
        y = torch.max(y, dim=2).values
        y = self.linear(y)
        y = torch.nn.functional.softmax(y, dim=-1)
        return y
