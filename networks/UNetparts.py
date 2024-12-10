import torch
from torch import nn

class DoubleConv(nn.Module):
    """
    Double convolution block

    Args:
    in_channels (int): Number of input channels
    out_channels (int): Number of output channels
    conv_op (nn.Sequential): Sequential convolution operation
    Conv2d (nn.Conv2d): Convolution layer
    ReLU (nn.ReLU): ReLU activation function
    Returns:
    nn.Module: DoubleConv block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass
        """
        return self.conv_op(x)
    
class DownSample(nn.Module):
    """
    Downsample block

    Args:
    in_channels (int): Number of input channels
    out_channels (int): Number of output channels
    conv (DoubleConv): Double convolution block
    pool (nn.MaxPool2d): Max pooling layer

    Returns:
    nn.Module: Downsample block
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the DownSample block
        """
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        """
        Forward pass of the DownSample block
        """
        down = self.conv(x)
        p = self.pool(down)
        return down, p

class upsample(nn.Module):
    """
    Upsample block

    Args:
    in_channels (int): Number of input channels
    out_channels (int): Number of output channels
    up (nn.ConvTranspose2d): Transposed convolution layer
    conv (DoubleConv): Double convolution block

    Returns:
    nn.Module: Upsample block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Forward Pass of the model
        """
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)
