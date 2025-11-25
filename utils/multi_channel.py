# utils/multi_channel.py
"""
多通道输入特征提取
raw + grad_xyz + LoG 是标配
LB 前7 全部使用
"""

import numpy as np
from scipy import ndimage
import torch


def compute_gradient_features(volume):
    """
    计算梯度特征
    
    Returns
    -------
    grad_x, grad_y, grad_z : np.ndarray
        三个方向的梯度
    """
    grad_z = np.gradient(volume, axis=0)
    grad_y = np.gradient(volume, axis=1)
    grad_x = np.gradient(volume, axis=2)
    
    return grad_x, grad_y, grad_z


def compute_log_features(volume, sigma=1.0):
    """
    计算 LoG (Laplacian of Gaussian) 特征
    
    Parameters
    ----------
    volume : np.ndarray
        输入 volume
    sigma : float
        高斯核标准差
        
    Returns
    -------
    log : np.ndarray
        LoG 特征
    """
    log = ndimage.gaussian_laplace(volume, sigma=sigma)
    return log


def compute_hessian_features(volume):
    """
    计算 Hessian 特征（可选，计算量大）
    
    Returns
    -------
    hxx, hyy, hzz : np.ndarray
        Hessian 矩阵的对角元素
    """
    grad_x, grad_y, grad_z = compute_gradient_features(volume)
    
    # 二阶导数
    hxx = np.gradient(grad_x, axis=2)
    hyy = np.gradient(grad_y, axis=1)
    hzz = np.gradient(grad_z, axis=0)
    
    return hxx, hyy, hzz


def compute_hessian_trace(volume):
    """计算 Hessian 迹（曲率相关特征，单通道）"""
    hxx, hyy, hzz = compute_hessian_features(volume)
    trace = hxx + hyy + hzz
    return trace


def compute_local_contrast(volume, kernel_size=5):
    """局部对比度/方差特征，用于增强局部纹理信息"""
    # 局部均值和均方
    footprint = np.ones((kernel_size, kernel_size, kernel_size), dtype=np.float32)
    mean = ndimage.uniform_filter(volume, footprint=footprint)
    mean_sq = ndimage.uniform_filter(volume ** 2, footprint=footprint)
    var = np.clip(mean_sq - mean ** 2, 0, None)
    contrast = np.sqrt(var + 1e-8)
    return contrast


def compute_multi_scale_log(volume, sigmas=(1.0, 2.0)):
    """多尺度 LoG 特征，返回若干尺度的 LoG 结果"""
    logs = []
    for s in sigmas:
        logs.append(ndimage.gaussian_laplace(volume, sigma=float(s)))
    return logs


def extract_multi_channel_features(volume, channels=['raw', 'grad', 'log']):
    """
    提取多通道特征
    
    Parameters
    ----------
    volume : np.ndarray
        输入 volume (D, H, W)
    channels : list
        要提取的通道类型
        - 'raw': 原始强度
        - 'grad': 梯度 (grad_x, grad_y, grad_z)
        - 'log': LoG
        - 'hessian': Hessian 对角元素
        
    Returns
    -------
    features : np.ndarray
        多通道特征 (C, D, H, W)
    """
    features = []
    
    # 原始强度
    if 'raw' in channels:
        features.append(volume)
    
    # 梯度
    if 'grad' in channels:
        grad_x, grad_y, grad_z = compute_gradient_features(volume)
        features.extend([grad_x, grad_y, grad_z])
    
    # LoG
    if 'log' in channels:
        log = compute_log_features(volume, sigma=1.0)
        features.append(log)

    # Multi-scale LoG
    if 'log_multi' in channels:
        multi_logs = compute_multi_scale_log(volume, sigmas=(1.0, 2.0))
        features.extend(multi_logs)
    
    # Hessian（可选）
    if 'hessian' in channels:
        hxx, hyy, hzz = compute_hessian_features(volume)
        features.extend([hxx, hyy, hzz])

    # Hessian 迹（单通道曲率特征）
    if 'hessian_trace' in channels:
        trace = compute_hessian_trace(volume)
        features.append(trace)

    # 局部对比度
    if 'local_contrast' in channels:
        lc = compute_local_contrast(volume, kernel_size=5)
        features.append(lc)
    
    # Stack to (C, D, H, W)
    features = np.stack(features, axis=0).astype(np.float32)
    
    # 归一化每个通道
    for i in range(features.shape[0]):
        channel = features[i]
        # 使用 percentile 归一化（更鲁棒）
        p1, p99 = np.percentile(channel, [1, 99])
        if p99 > p1:
            channel = (channel - p1) / (p99 - p1)
            channel = np.clip(channel, 0, 1)
        features[i] = channel
    
    return features


class MultiChannelVesuviusDataset:
    """
    多通道 Vesuvius 数据集包装器
    
    使用方法：
    ```python
    from utils.ink_sampling import InkAwareVesuviusDataset
    from utils.multi_channel import MultiChannelVesuviusDataset
    
    base_dataset = InkAwareVesuviusDataset(...)
    dataset = MultiChannelVesuviusDataset(
        base_dataset,
        channels=['raw', 'grad', 'log']
    )
    ```
    """
    
    def __init__(self, base_dataset, channels=['raw', 'grad']):
        """
        Parameters
        ----------
        base_dataset : Dataset
            基础数据集（如 InkAwareVesuviusDataset）
        channels : list
            要提取的通道
        """
        self.base_dataset = base_dataset
        self.channels = channels
        
        # 计算通道数
        self.num_channels = 0
        if 'raw' in channels:
            self.num_channels += 1
        if 'grad' in channels:
            self.num_channels += 3
        if 'log' in channels:
            self.num_channels += 1
        if 'log_multi' in channels:
            # 默认两尺度 LoG
            self.num_channels += 2
        if 'hessian' in channels:
            self.num_channels += 3
        if 'hessian_trace' in channels:
            self.num_channels += 1
        if 'local_contrast' in channels:
            self.num_channels += 1
        
        print(f"🎨 多通道特征:")
        print(f"   通道配置: {channels}")
        print(f"   总通道数: {self.num_channels}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """获取多通道样本"""
        # 从基础数据集获取
        volume, mask = self.base_dataset[idx]
        
        # volume: (1, D, H, W) -> (D, H, W)
        volume_np = volume.squeeze(0).numpy()
        
        # 提取多通道特征
        features = extract_multi_channel_features(volume_np, self.channels)
        
        # 转换为 tensor
        features = torch.from_numpy(features).float()  # (C, D, H, W)
        
        return features, mask


def get_channel_count(channels):
    """获取通道数"""
    count = 0
    if 'raw' in channels:
        count += 1
    if 'grad' in channels:
        count += 3
    if 'log' in channels:
        count += 1
    if 'log_multi' in channels:
        count += 2
    if 'hessian' in channels:
        count += 3
    if 'hessian_trace' in channels:
        count += 1
    if 'local_contrast' in channels:
        count += 1
    return count


if __name__ == "__main__":
    print("="*60)
    print("测试多通道特征提取")
    print("="*60)
    
    # 创建测试数据
    volume = np.random.rand(64, 64, 64).astype(np.float32)
    
    print(f"\n输入 Volume: {volume.shape}")
    
    # 测试不同通道组合
    test_configs = [
        ['raw'],
        ['raw', 'grad'],
        ['raw', 'grad', 'log'],
        ['raw', 'grad', 'log', 'hessian'],
        ['raw', 'grad', 'log', 'log_multi'],
        ['raw', 'grad', 'log', 'hessian_trace', 'local_contrast'],
    ]
    
    for channels in test_configs:
        features = extract_multi_channel_features(volume, channels)
        print(f"\n通道配置: {channels}")
        print(f"  输出形状: {features.shape}")
        print(f"  通道数: {features.shape[0]}")
        print(f"  范围: [{features.min():.3f}, {features.max():.3f}]")
    
    print("\n✅ 测试通过！")
    print("="*60)
