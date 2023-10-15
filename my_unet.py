import torch
import torch.nn as nn
"""
这个UNet写得过于丑陋，而且出来的效果并不好，图片大小也不对
所以纯当自己的一个练手项目，不必参考，
最后参考网络教程重新写了一个UNet，见unet.py
"""

# 搭建网络用于预测噪声
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # 编码器（下采样路径）
        self.down_conv0 = nn.Conv2d(in_channels, 64, kernel_size=3)
        self.down_conv1 = nn.Conv2d(64, 64, kernel_size=3)
        self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.down_conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.down_conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.down_conv5 = nn.Conv2d(256, 256, kernel_size=3)
        self.down_conv6 = nn.Conv2d(256, 512, kernel_size=3)
        self.down_conv7 = nn.Conv2d(512, 512, kernel_size=3)
        self.down_conv8 = nn.Conv2d(512, 1024, kernel_size=3)
        self.down_conv9 = nn.Conv2d(1024, 1024, kernel_size=3)
        self.maxpool = nn.MaxPool2d(2)

        # 解码器（上采样路径）
        self.up_conv0 = nn.Conv2d(1024, 512, kernel_size=3)
        self.up_conv1 = nn.Conv2d(512, 512, kernel_size=3)
        self.up_conv2 = nn.Conv2d(512, 256, kernel_size=3)
        self.up_conv3 = nn.Conv2d(256, 256, kernel_size=3)
        self.up_conv4 = nn.Conv2d(256, 128, kernel_size=3)
        self.up_conv5 = nn.Conv2d(128, 128, kernel_size=3)
        self.up_conv6 = nn.Conv2d(128, 64, kernel_size=3)
        self.up_conv7 = nn.Conv2d(64, 64, kernel_size=3)
        self.up_conv8 = nn.Conv2d(64, out_channels, kernel_size=1)

        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def down_sample(self, x):
        x1 = self.relu(self.down_conv0(x))
        x2 = self.relu(self.down_conv1(x1))
        x3 = self.maxpool(x2)
        x4 = self.relu(self.down_conv2(x3))
        x5 = self.relu(self.down_conv3(x4))
        x6 = self.maxpool(x5)
        x7 = self.relu(self.down_conv4(x6))
        x8 = self.relu(self.down_conv5(x7))
        x9 = self.maxpool(x8)
        x10 = self.relu(self.down_conv6(x9))
        x11 = self.relu(self.down_conv7(x10))
        x12 = self.maxpool(x11)
        x13 = self.relu(self.down_conv8(x12))
        x14 = self.relu(self.down_conv9(x13))
        return x14, x11, x8, x5, x2

    def up_sample(self, x, x11, x8, x5, x2):
        """
        在上采样过程中，我参照原论文中图示，并用自己的想法进行实现
        先使用1x1的卷积核使当前结果的通道数减半，
        再使用双线性插值进行上采样，
        将下采样结果进行裁剪后拼接
        """
        def crop_tensor(target, source):
            target_size = target.shape[2:]
            source_size = source.shape[2:]
            delta_h = source_size[0] - target_size[0]
            delta_w = source_size[1] - target_size[1]

            if delta_h > 0:
                source = source[:, :, delta_h // 2:-delta_h // 2, :]
            if delta_w > 0:
                source = source[:, :, :, delta_w // 2:-delta_w // 2]
            return source
        
        x = self.upsample(nn.Conv2d(x.shape[1], x.shape[1]//2, 1)(x))
        x = self.relu(self.up_conv0(torch.cat([x, crop_tensor(x, x11)], dim=1)))
        x = self.relu(self.up_conv1(x))
        x = self.upsample(nn.Conv2d(x.shape[1], x.shape[1]//2, 1)(x))
        x = self.relu(self.up_conv2(torch.cat([x, crop_tensor(x, x8)], dim=1)))
        x = self.relu(self.up_conv3(x))
        x = self.upsample(nn.Conv2d(x.shape[1], x.shape[1]//2, 1)(x))
        x = self.relu(self.up_conv4(torch.cat([x, crop_tensor(x, x5)], dim=1)))
        x = self.relu(self.up_conv5(x))
        x = self.upsample(nn.Conv2d(x.shape[1], x.shape[1]//2, 1)(x))
        x = self.relu(self.up_conv6(torch.cat([x, crop_tensor(x, x2)], dim=1)))
        x = self.relu(self.up_conv7(x))
        x = self.relu(self.up_conv8(x))
        return x

    def forward(self, x):
        x = self.up_sample(*self.down_sample(x))
        return x
    

if __name__ == '__main__':
    # 测试网络结构
    net = UNet(3, 3)
    x = torch.randn(1, 3, 572, 572)
    y = net(x)
    print(y.shape)