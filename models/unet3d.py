"""
3D U-Net 模型

用于 3D 医学图像分割
优化用于 RTX 5090 (24GB VRAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """双卷积层"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """下采样层"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """上采样层"""
    
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        
        # in_channels: 来自下层的通道数
        # out_channels: skip connection 的通道数（也是最终输出通道数）
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            # trilinear: 上采样不改变通道数，concat 后是 in_channels + out_channels
            self.conv = DoubleConv3D(in_channels + out_channels, out_channels)
        else:
            # ConvTranspose: 上采样同时减半通道数，concat 后是 in_channels//2 + out_channels = in_channels
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理尺寸不匹配
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net 模型
    
    Parameters
    ----------
    in_channels : int
        输入通道数
    out_channels : int
        输出通道数
    base_channels : int
        基础通道数（默认 32，可减小以节省显存）
    trilinear : bool
        是否使用三线性插值上采样
    """
    
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, trilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trilinear = trilinear
        
        # Encoder
        self.inc = DoubleConv3D(in_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)
        factor = 2 if trilinear else 1
        self.down4 = Down3D(base_channels * 8, base_channels * 16 // factor)
        
        # Decoder
        # Up3D(in_channels, out_channels): in_channels 是来自下层的通道数，out_channels 是 skip connection 的通道数
        # concat 后会是 in_channels + out_channels，然后卷积到 out_channels
        self.up1 = Up3D(base_channels * 16 // factor, base_channels * 8, trilinear)
        self.up2 = Up3D(base_channels * 8, base_channels * 4, trilinear)
        self.up3 = Up3D(base_channels * 4, base_channels * 2, trilinear)
        self.up4 = Up3D(base_channels * 2, base_channels, trilinear)
        
        # Output
        self.outc = nn.Conv3d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits


class UNet3DLite(nn.Module):
    """
    轻量级 3D U-Net
    
    显存优化版本，适合 RTX 5090 24GB
    - 更少的通道数
    - 更少的下采样层
    - 适合大 patch size
    """
    
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder (3 levels)
        self.inc = DoubleConv3D(in_channels, 16)
        self.down1 = Down3D(16, 32)
        self.down2 = Down3D(32, 64)
        self.down3 = Down3D(64, 128)
        
        # Decoder
        self.up1 = Up3D(128, 64, trilinear=True)
        self.up2 = Up3D(64, 32, trilinear=True)
        self.up3 = Up3D(32, 16, trilinear=True)
        
        # Output
        self.outc = nn.Conv3d(16, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits


def test_model():
    """测试模型"""
    print("测试 3D U-Net 模型...")
    
    # 创建模型
    model = UNet3D(in_channels=1, out_channels=1, base_channels=32)
    model_lite = UNet3DLite(in_channels=1, out_channels=1)
    
    # 测试输入
    x = torch.randn(1, 1, 64, 64, 64)  # (B, C, D, H, W)
    
    # 前向传播
    with torch.no_grad():
        y = model(x)
        y_lite = model_lite(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状 (UNet3D): {y.shape}")
    print(f"输出形状 (UNet3DLite): {y_lite.shape}")
    
    # 计算参数量
    params = sum(p.numel() for p in model.parameters())
    params_lite = sum(p.numel() for p in model_lite.parameters())
    
    print(f"\n参数量:")
    print(f"  UNet3D: {params:,} ({params/1e6:.2f}M)")
    print(f"  UNet3DLite: {params_lite:,} ({params_lite/1e6:.2f}M)")
    
    # 估算显存
    print(f"\n估算显存 (batch_size=2, patch=64³):")
    print(f"  UNet3D: ~18-20 GB")
    print(f"  UNet3DLite: ~12-14 GB")


if __name__ == "__main__":
    test_model()
