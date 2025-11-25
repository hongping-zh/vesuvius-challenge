"""
Vesuvius Challenge Inference Notebook

用于 Kaggle Notebook 提交
- CPU only
- 9 小时时间限制
- 从 Kaggle Dataset 加载模型
- 滑动窗口推理
- 拓扑感知后处理
- 生成提交文件
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import time
from tqdm import tqdm
import pandas as pd


# ============================================================================
# 模型定义（复制自 models/unet3d.py）
# ============================================================================

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
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels + out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理尺寸不匹配
        import torch.nn.functional as F
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net 模型"""
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


# ============================================================================
# 后处理（简化版）
# ============================================================================

from scipy import ndimage
from skimage.morphology import remove_small_objects, remove_small_holes


def postprocess_prediction(pred, min_size=100, min_hole_size=50):
    """
    简化的后处理
    
    Parameters
    ----------
    pred : np.ndarray
        预测掩码 (D, H, W)
    min_size : int
        最小组件大小
    min_hole_size : int
        最小孔洞大小
    
    Returns
    -------
    np.ndarray
        后处理后的掩码
    """
    # 转换为布尔
    pred = pred.astype(bool)
    
    # 移除小组件
    pred = remove_small_objects(pred, min_size=min_size, connectivity=2)
    
    # 填充小孔洞
    pred = remove_small_holes(pred, area_threshold=min_hole_size, connectivity=2)
    
    return pred.astype(np.float32)


# ============================================================================
# 滑动窗口推理
# ============================================================================

def sliding_window_inference(
    model,
    volume,
    patch_size=(64, 64, 64),
    overlap=0.5,
    batch_size=4,
    device='cpu'
):
    """
    滑动窗口推理
    
    Parameters
    ----------
    model : nn.Module
        训练好的模型
    volume : np.ndarray
        输入体积 (D, H, W)
    patch_size : tuple
        Patch 大小
    overlap : float
        重叠比例
    batch_size : int
        批次大小
    device : str
        设备
    
    Returns
    -------
    np.ndarray
        预测结果 (D, H, W)
    """
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    
    # 计算步长
    stride_d = int(pd * (1 - overlap))
    stride_h = int(ph * (1 - overlap))
    stride_w = int(pw * (1 - overlap))
    
    # 生成 patch 坐标
    patches = []
    for d in range(0, D - pd + 1, stride_d):
        for h in range(0, H - ph + 1, stride_h):
            for w in range(0, W - pw + 1, stride_w):
                patches.append((d, h, w))
    
    print(f"总共 {len(patches)} 个 patches")
    
    # 输出累积
    output = np.zeros((D, H, W), dtype=np.float32)
    counts = np.zeros((D, H, W), dtype=np.float32)
    
    # 批次推理
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="推理中"):
            batch_patches = patches[i:i + batch_size]
            
            # 提取 patches
            batch_data = []
            for d, h, w in batch_patches:
                patch = volume[d:d+pd, h:h+ph, w:w+pw]
                batch_data.append(patch)
            
            # 转换为 tensor
            batch_tensor = torch.from_numpy(np.array(batch_data)).float()
            batch_tensor = batch_tensor.unsqueeze(1)  # (B, 1, D, H, W)
            batch_tensor = batch_tensor.to(device)
            
            # 推理
            pred = model(batch_tensor)
            pred = torch.sigmoid(pred)
            pred = pred.cpu().numpy()
            
            # 累积结果
            for j, (d, h, w) in enumerate(batch_patches):
                output[d:d+pd, h:h+ph, w:w+pw] += pred[j, 0]
                counts[d:d+pd, h:h+ph, w:w+pw] += 1
    
    # 平均
    output = output / (counts + 1e-8)
    
    return output


# ============================================================================
# 主推理流程
# ============================================================================

def main():
    """主推理函数"""
    print("=" * 60)
    print("Vesuvius Challenge Inference")
    print("=" * 60)
    print()
    
    start_time = time.time()
    
    # ========================================
    # 1. 加载模型
    # ========================================
    print("📥 加载模型...")
    
    # Kaggle Dataset 路径（需要修改为实际路径）
    model_path = '/kaggle/input/vesuvius-model/best_model.pth'
    
    # 如果本地测试，使用本地路径
    if not Path(model_path).exists():
        model_path = 'models/checkpoints/best_model.pth'
    
    # 创建模型
    model = UNet3D(in_channels=1, out_channels=1, base_channels=48)
    
    # 加载权重（如果存在）
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 模型加载完成")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best Score: {checkpoint.get('best_dice', 'N/A'):.4f}")
    else:
        print(f"⚠️  模型文件不存在，使用随机初始化权重")
        print(f"   (仅用于测试推理流程)")
    
    model.eval()
    
    # ========================================
    # 2. 加载测试数据
    # ========================================
    print("\n📥 加载测试数据...")
    
    # Kaggle 测试数据路径
    test_path = '/kaggle/input/vesuvius-challenge-surface-detection/test/volume.zarr'
    
    # 如果本地测试
    if not Path(test_path).exists():
        test_path = 'data/processed/test/volume.npy'
    
    # 加载数据
    if test_path.endswith('.zarr'):
        import zarr
        volume = zarr.open(test_path, mode='r')
        volume = np.array(volume)
    else:
        volume = np.load(test_path)
    
    volume = volume.astype(np.float32)
    
    print(f"✓ 测试数据加载完成")
    print(f"  形状: {volume.shape}")
    print(f"  范围: [{volume.min():.4f}, {volume.max():.4f}]")
    
    # 归一化
    mean = volume.mean()
    std = volume.std()
    volume = (volume - mean) / (std + 1e-8)
    
    # ========================================
    # 3. 推理
    # ========================================
    print("\n🔮 开始推理...")
    
    predictions = sliding_window_inference(
        model=model,
        volume=volume,
        patch_size=(80, 80, 80),
        overlap=0.5,
        batch_size=4,
        device='cpu'
    )
    
    print(f"✓ 推理完成")
    print(f"  预测范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # ========================================
    # 4. 后处理
    # ========================================
    print("\n🔧 后处理...")
    
    # 二值化
    predictions_binary = (predictions > 0.5).astype(np.float32)
    
    # 拓扑修正
    predictions_final = postprocess_prediction(
        predictions_binary,
        min_size=100,
        min_hole_size=50
    )
    
    print(f"✓ 后处理完成")
    print(f"  预测像素: {predictions_final.sum():.0f} / {predictions_final.size}")
    print(f"  覆盖率: {predictions_final.mean() * 100:.2f}%")
    
    # ========================================
    # 5. 生成提交文件
    # ========================================
    print("\n📤 生成提交文件...")
    
    # 创建提交 DataFrame（根据比赛要求调整格式）
    # 这里是示例格式，需要根据实际比赛要求修改
    submission = pd.DataFrame({
        'id': ['sample_id'],  # 替换为实际 ID
        'prediction': [predictions_final.flatten().tolist()]  # 或其他格式
    })
    
    submission.to_csv('submission.csv', index=False)
    
    print(f"✓ 提交文件已生成: submission.csv")
    
    # ========================================
    # 总结
    # ========================================
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    
    print("\n" + "=" * 60)
    print("✅ 推理完成！")
    print("=" * 60)
    print(f"总耗时: {hours}h {minutes}m")
    print(f"时间限制: 9h (剩余: {9 - hours}h {60 - minutes}m)")
    
    if elapsed_time < 9 * 3600:
        print("✓ 在时间限制内完成")
    else:
        print("⚠️ 超过时间限制！")


if __name__ == "__main__":
    main()
