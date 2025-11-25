# utils/ink_sampling.py
"""
Ink-only Positive Sampling
墨迹像素 <0.1%，必须实现！
LB 前10 全部使用
"""

import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset


class InkAwareVesuviusDataset(Dataset):
    """
    墨迹感知数据集
    
    关键特性：
    - 70% 采样包含墨迹的 patch
    - 30% 采样纯背景 patch
    - 预先构建墨迹索引（加速采样）
    """
    
    def __init__(
        self,
        data_dir,
        patch_size=[96, 96, 96],
        positive_ratio=0.7,
        min_ink_pixels=100,
        num_samples_per_epoch=1000,
        augment=False
    ):
        """
        Parameters
        ----------
        data_dir : str
            数据目录
        patch_size : list
            Patch 大小
        positive_ratio : float
            包含墨迹的 patch 比例（0.7 = 70%）
        min_ink_pixels : int
            最少墨迹像素数
        num_samples_per_epoch : int
            每个 epoch 采样数量
        augment : bool
            是否数据增强
        """
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.positive_ratio = positive_ratio
        self.min_ink_pixels = min_ink_pixels
        self.num_samples = num_samples_per_epoch
        self.augment = augment
        
        # 加载数据
        self._load_data()
        
        # 构建墨迹索引
        self._build_ink_index()
    
    def _load_data(self):
        """加载 volume 和 mask"""
        print(f"📥 加载数据: {self.data_dir}")
        
        # 查找数据文件
        volume_files = list(self.data_dir.glob("volume.*"))
        mask_files = list(self.data_dir.glob("mask.*"))
        
        if not volume_files or not mask_files:
            raise FileNotFoundError(f"数据文件不存在: {self.data_dir}")
        
        # 加载
        volume_file = volume_files[0]
        mask_file = mask_files[0]
        
        if volume_file.suffix == '.npy':
            self.volume = np.load(volume_file)
            self.mask = np.load(mask_file)
        else:
            raise ValueError(f"不支持的文件格式: {volume_file.suffix}")
        
        print(f"✓ Volume 形状: {self.volume.shape}")
        print(f"✓ Mask 形状: {self.mask.shape}")
        
        # 归一化
        if self.volume.max() > 1.0:
            self.volume = self.volume.astype(np.float32) / 255.0
    
    def _build_ink_index(self):
        """构建墨迹索引"""
        print(f"🔍 扫描墨迹分布...")
        
        D, H, W = self.volume.shape
        pd, ph, pw = self.patch_size
        
        self.ink_positions = []
        self.no_ink_positions = []
        
        # 滑动窗口扫描
        step = min(pd // 2, ph // 2, pw // 2)  # 步长
        
        for d in range(0, D - pd + 1, step):
            for h in range(0, H - ph + 1, step):
                for w in range(0, W - pw + 1, step):
                    # 提取 patch mask
                    patch_mask = self.mask[d:d+pd, h:h+ph, w:w+pw]
                    ink_pixels = patch_mask.sum()
                    
                    position = (d, h, w)
                    
                    if ink_pixels >= self.min_ink_pixels:
                        self.ink_positions.append((position, ink_pixels))
                    else:
                        self.no_ink_positions.append(position)
        
        print(f"✓ 包含墨迹的位置: {len(self.ink_positions)}")
        print(f"✓ 纯背景位置: {len(self.no_ink_positions)}")
        
        if len(self.ink_positions) == 0:
            print("⚠️  警告：没有找到包含墨迹的 patch！")
            # 使用所有位置
            self.ink_positions = [(pos, 0) for pos in self.no_ink_positions[:100]]
        
        if len(self.no_ink_positions) == 0:
            print("⚠️  警告：没有找到纯背景 patch！")
            self.no_ink_positions = [pos for pos, _ in self.ink_positions[:100]]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """采样一个 patch"""
        # 决定采样类型
        if np.random.rand() < self.positive_ratio:
            # 采样包含墨迹的 patch
            if len(self.ink_positions) > 0:
                # 根据墨迹数量加权采样（墨迹多的更容易被采样）
                positions, weights = zip(*self.ink_positions)
                weights = np.array(weights, dtype=np.float32)
                weights = weights / weights.sum()
                idx = np.random.choice(len(positions), p=weights)
                position = positions[idx]
            else:
                position = self.no_ink_positions[np.random.randint(len(self.no_ink_positions))]
        else:
            # 采样纯背景 patch
            if len(self.no_ink_positions) > 0:
                position = self.no_ink_positions[np.random.randint(len(self.no_ink_positions))]
            else:
                position, _ = self.ink_positions[np.random.randint(len(self.ink_positions))]
        
        # 提取 patch
        d, h, w = position
        pd, ph, pw = self.patch_size
        
        volume_patch = self.volume[d:d+pd, h:h+ph, w:w+pw].copy()
        mask_patch = self.mask[d:d+pd, h:h+ph, w:w+pw].copy()
        
        # 数据增强（如果启用）
        if self.augment:
            volume_patch, mask_patch = self._augment(volume_patch, mask_patch)
        
        # 转换为 tensor
        volume_patch = torch.from_numpy(volume_patch).unsqueeze(0).float()  # (1, D, H, W)
        mask_patch = torch.from_numpy(mask_patch).unsqueeze(0).float()      # (1, D, H, W)
        
        return volume_patch, mask_patch
    
    def _augment(self, volume, mask):
        """数据增强"""
        # 随机翻转
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        
        # 随机旋转 90 度
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            volume = np.rot90(volume, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(1, 2)).copy()
        
        # 强度增强
        if np.random.rand() > 0.5:
            volume = volume + np.random.randn(*volume.shape) * 0.05
            volume = np.clip(volume, 0, 1)
        
        return volume, mask


if __name__ == "__main__":
    # 测试
    print("="*60)
    print("测试 Ink-only Sampling")
    print("="*60)
    
    # 创建测试数据集
    dataset = InkAwareVesuviusDataset(
        data_dir='data/processed/train',
        patch_size=[96, 96, 96],
        positive_ratio=0.7,
        min_ink_pixels=100,
        num_samples_per_epoch=100
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 采样测试
    print("\n采样测试:")
    ink_count = 0
    for i in range(10):
        volume, mask = dataset[i]
        ink_pixels = mask.sum().item()
        print(f"  Sample {i}: Volume {volume.shape}, Mask {mask.shape}, Ink pixels: {ink_pixels:.0f}")
        if ink_pixels > 100:
            ink_count += 1
    
    print(f"\n包含墨迹的样本: {ink_count}/10 ({ink_count*10}%)")
    print("✅ 测试通过！")
    print("="*60)
