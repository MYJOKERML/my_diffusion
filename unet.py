import torch
import torch.nn as nn

"""
重新搭了个UNet，
卷积操作增加了标准化，
上采样操作使用了反卷积，
最后出来的图像和原图像尺寸相同
参考链接：https://zhuanlan.zhihu.com/p/609396803
"""

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        in_channels: 输入通道数
        out_channels: 输出通道数
        """
        super(UNet, self).__init__()

        # 编码器（下采样路径）
        self.down_conv1 = nn.Sequential(
            self.conv_block(in_channels, 64),
            nn.MaxPool2d(kernel_size=2),
        )
        self.down_conv2 = nn.Sequential(
            self.conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2),
        )
        self.down_conv3 = nn.Sequential(
            self.conv_block(128, 256),
            nn.MaxPool2d(kernel_size=2),
        )
        self.down_conv4 = nn.Sequential(
            self.conv_block(256, 512),
            nn.MaxPool2d(kernel_size=2),
        )
        self.down_conv5 = nn.Sequential(
            self.conv_block(512, 1024),
            nn.MaxPool2d(kernel_size=2),
        )

        # 解码器（上采样路径）
        self.up_sample1 = self.conv_transpose_block(1024, 512)
        self.up_conv1 = self.conv_block(1024, 512)
        self.up_sample2 = self.conv_transpose_block(512, 256)
        self.up_conv2 = self.conv_block(512, 256)
        self.up_sample3 = self.conv_transpose_block(256, 128)
        self.up_conv3 = self.conv_block(256, 128)
        self.up_sample4 = self.conv_transpose_block(128, 64)
        self.up_conv4 = self.conv_block(128, 64)

        self.restore_shape_layer = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        
        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)
    

    def forward(self, x, t):
        # 下采样
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)
        x4 = self.down_conv4(x3)
        x = self.down_conv5(x4)
        """
        x1: torch.Size([1, 64, 128, 128]),
        x2: torch.Size([1, 128, 64, 64]),
        x3: torch.Size([1, 256, 32, 32]),
        x4: torch.Size([1, 512, 16, 16]),
        x: torch.Size([1, 1024, 8, 8])
        """

        # 上采样
        x = self.up_sample1(x)
        x = torch.cat([x, x4], dim=1)
        x = self.up_conv1(x)
        x = self.up_sample2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv2(x)
        x = self.up_sample3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv3(x)
        x = self.up_sample4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv4(x)
        x = self.restore_shape_layer(x)
        x = self.output_layer(x)

        return x

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        return block
    
    def conv_transpose_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
        return block