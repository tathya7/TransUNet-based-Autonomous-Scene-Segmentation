import torch
import torch.nn as nn
from networks.UNetparts import DoubleConv, DownSample, upsample

class IUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
            super().__init__()
            self.down_conv_1 = DownSample(in_channels, 64)
            self.down_conv_2 = DownSample(64, 128)
            self.down_conv_3 = DownSample(128, 256)
            self.down_conv_4 = DownSample(256, 512)

            self.bottleneck = DoubleConv(512, 1024)

            self.up_conv_1 = upsample(1024, 512)
            self.up_conv_2 = upsample(512, 256)
            self.up_conv_3 = upsample(256, 128)
            self.up_conv_4 = upsample(128, 64)

            self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
        #     self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_conv_1(x)
        down_2, p2 = self.down_conv_2(p1)
        down_3, p3 = self.down_conv_3(p2)
        down_4, p4 = self.down_conv_4(p3)

        b = self.bottleneck(p4)

        up_1 = self.up_conv_1(b, down_4)
        up_2 = self.up_conv_2(up_1, down_3)
        up_3 = self.up_conv_3(up_2, down_2)
        up_4 = self.up_conv_4(up_3, down_1)

        out = self.out(up_4)
        return out

# if __name__ == "__main__":
#     double_conv = DoubleConv(256, 256)
#     print(double_conv)

#     input = torch.rand((1, 3, 512, 512))
#     model = UNet(3, 10)
#     output = model(input)
#     print(output.size())


