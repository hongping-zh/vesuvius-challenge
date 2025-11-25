"""
Vesuvius Challenge 数据集加载器

处理 3D CT 扫描数据:
- 二值标注的 CT 扫描块
- 来自 ESRF (Grenoble) 和 DLS (Oxford) 同步加速器
- 支持 .zarr, .tif, .npy 格式
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import zarr
from PIL import Image
import cv2
from scipy import ndimage
import albumentations as A


class VesuviusDataset(Dataset):
    """
    Vesuvius Challenge 数据集
    
    处理 3D CT 扫描体积数据
    """
    
    def __init__(
        self,
        data_dir,
        patch_size=(64, 64, 64),
        augment=False,
        normalize=True
    ):
        """
        Parameters
        ----------
        data_dir : str
            数据目录路径
        patch_size : tuple
            3D patch 大小 (D, H, W)
        augment : bool
            是否使用数据增强
        normalize : bool
            是否归一化
        """
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.normalize = normalize
        
        # 加载数据文件列表
        self.volume_files = []
        self.mask_files = []
        
        self._load_file_list()
        
        # 数据增强
        if self.augment:
            self.transform = self._get_augmentation()
        else:
            self.transform = None
    
    def _load_file_list(self):
        """加载数据文件列表"""
        # 查找体积文件
        for volume_file in self.data_dir.glob("**/volume.*"):
            # 查找对应的掩码文件
            mask_file = volume_file.parent / f"mask{volume_file.suffix}"
            
            if mask_file.exists():
                self.volume_files.append(volume_file)
                self.mask_files.append(mask_file)
        
        if len(self.volume_files) == 0:
            print(f"⚠️ 警告: 在 {self.data_dir} 中未找到数据文件")
            print("预期文件结构:")
            print("  data/")
            print("    train/")
            print("      volume.zarr  或  volume.npy")
            print("      mask.zarr    或  mask.npy")
        else:
            print(f"✓ 找到 {len(self.volume_files)} 个数据文件")
    
    def _get_augmentation(self):
        """获取数据增强"""
        # 3D 数据增强
        return {
            'flip': True,
            'rotate': True,
            'elastic': True,
            'intensity': True
        }
    
    def _load_volume(self, file_path):
        """
        加载体积数据
        
        支持格式: .zarr, .npy, .tif
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.zarr':
            # Zarr 格式（推荐）
            volume = zarr.open(str(file_path), mode='r')
            volume = np.array(volume)
        
        elif file_path.suffix == '.npy':
            # NumPy 格式
            volume = np.load(str(file_path))
        
        elif file_path.suffix in ['.tif', '.tiff']:
            # TIFF 堆栈
            from tifffile import imread
            volume = imread(str(file_path))
        
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        return volume
    
    def _extract_random_patch(self, volume, mask):
        """
        提取随机 3D patch
        
        Parameters
        ----------
        volume : np.ndarray
            完整体积 (D, H, W)
        mask : np.ndarray
            完整掩码 (D, H, W)
        
        Returns
        -------
        tuple
            (volume_patch, mask_patch)
        """
        D, H, W = volume.shape
        pd, ph, pw = self.patch_size
        
        # 随机起始位置
        d_start = np.random.randint(0, max(1, D - pd))
        h_start = np.random.randint(0, max(1, H - ph))
        w_start = np.random.randint(0, max(1, W - pw))
        
        # 提取 patch
        volume_patch = volume[
            d_start:d_start + pd,
            h_start:h_start + ph,
            w_start:w_start + pw
        ]
        
        mask_patch = mask[
            d_start:d_start + pd,
            h_start:h_start + ph,
            w_start:w_start + pw
        ]
        
        return volume_patch, mask_patch
    
    def _augment_3d(self, volume, mask):
        """
        3D 数据增强
        
        Parameters
        ----------
        volume : np.ndarray
            体积数据 (D, H, W)
        mask : np.ndarray
            掩码数据 (D, H, W)
        
        Returns
        -------
        tuple
            增强后的 (volume, mask)
        """
        if not self.augment:
            return volume, mask
        
        # 随机翻转
        if self.transform['flip'] and np.random.rand() > 0.5:
            axis = np.random.choice([0, 1, 2])
            volume = np.flip(volume, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()
        
        # 随机旋转（在 xy 平面）
        if self.transform['rotate'] and np.random.rand() > 0.5:
            k = np.random.randint(1, 4)  # 90, 180, 270 度
            volume = np.rot90(volume, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
        
        # 强度变换
        if self.transform['intensity'] and np.random.rand() > 0.5:
            # 随机亮度和对比度
            alpha = np.random.uniform(0.8, 1.2)  # 对比度
            beta = np.random.uniform(-0.1, 0.1)  # 亮度
            volume = np.clip(volume * alpha + beta, 0, 1)
        
        # 弹性变形（简化版）
        if self.transform['elastic'] and np.random.rand() > 0.7:
            # 对每个 2D 切片应用弹性变形
            for d in range(volume.shape[0]):
                if np.random.rand() > 0.5:
                    # 使用 scipy 的 map_coordinates 实现简单变形
                    dx = np.random.randn(*volume[d].shape) * 2
                    dy = np.random.randn(*volume[d].shape) * 2
                    
                    x, y = np.meshgrid(
                        np.arange(volume.shape[1]),
                        np.arange(volume.shape[2]),
                        indexing='ij'
                    )
                    
                    indices = np.array([x + dx, y + dy])
                    
                    volume[d] = ndimage.map_coordinates(
                        volume[d], indices, order=1, mode='reflect'
                    )
                    mask[d] = ndimage.map_coordinates(
                        mask[d], indices, order=0, mode='reflect'
                    )
        
        return volume, mask
    
    def _normalize(self, volume):
        """
        归一化体积数据
        
        Parameters
        ----------
        volume : np.ndarray
            体积数据
        
        Returns
        -------
        np.ndarray
            归一化后的数据
        """
        if not self.normalize:
            return volume
        
        # Z-score 归一化
        mean = volume.mean()
        std = volume.std()
        
        if std > 0:
            volume = (volume - mean) / std
        
        return volume
    
    def __len__(self):
        """数据集大小"""
        return len(self.volume_files)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        Parameters
        ----------
        idx : int
            样本索引
        
        Returns
        -------
        tuple
            (volume, mask) 张量
        """
        # 加载数据
        volume = self._load_volume(self.volume_files[idx])
        mask = self._load_volume(self.mask_files[idx])
        
        # 确保是浮点数
        volume = volume.astype(np.float32)
        mask = mask.astype(np.float32)
        
        # 提取 patch
        volume_patch, mask_patch = self._extract_random_patch(volume, mask)
        
        # 数据增强
        volume_patch, mask_patch = self._augment_3d(volume_patch, mask_patch)
        
        # 归一化
        volume_patch = self._normalize(volume_patch)
        
        # 转换为 PyTorch 张量
        volume_tensor = torch.from_numpy(volume_patch).unsqueeze(0)  # (1, D, H, W)
        mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0)  # (1, D, H, W)
        
        return volume_tensor, mask_tensor


class VesuviusInferenceDataset(Dataset):
    """
    推理数据集
    
    用于完整体积的推理
    """
    
    def __init__(
        self,
        volume_path,
        patch_size=(64, 64, 64),
        overlap=0.5,
        normalize=True
    ):
        """
        Parameters
        ----------
        volume_path : str
            体积文件路径
        patch_size : tuple
            Patch 大小
        overlap : float
            重叠比例 [0, 1)
        normalize : bool
            是否归一化
        """
        self.volume_path = Path(volume_path)
        self.patch_size = patch_size
        self.overlap = overlap
        self.normalize = normalize
        
        # 加载体积
        self.volume = self._load_volume(self.volume_path)
        
        # 生成 patch 坐标
        self.patch_coords = self._generate_patch_coords()
        
        print(f"✓ 加载体积: {self.volume.shape}")
        print(f"✓ 生成 {len(self.patch_coords)} 个 patches")
    
    def _load_volume(self, file_path):
        """加载体积数据"""
        file_path = Path(file_path)
        
        if file_path.suffix == '.zarr':
            volume = zarr.open(str(file_path), mode='r')
            volume = np.array(volume)
        elif file_path.suffix == '.npy':
            volume = np.load(str(file_path))
        elif file_path.suffix in ['.tif', '.tiff']:
            from tifffile import imread
            volume = imread(str(file_path))
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        return volume.astype(np.float32)
    
    def _generate_patch_coords(self):
        """生成 patch 坐标"""
        D, H, W = self.volume.shape
        pd, ph, pw = self.patch_size
        
        # 计算步长
        stride_d = int(pd * (1 - self.overlap))
        stride_h = int(ph * (1 - self.overlap))
        stride_w = int(pw * (1 - self.overlap))
        
        coords = []
        
        for d in range(0, D - pd + 1, stride_d):
            for h in range(0, H - ph + 1, stride_h):
                for w in range(0, W - pw + 1, stride_w):
                    coords.append((d, h, w))
        
        return coords
    
    def _normalize(self, volume):
        """归一化"""
        if not self.normalize:
            return volume
        
        mean = volume.mean()
        std = volume.std()
        
        if std > 0:
            volume = (volume - mean) / std
        
        return volume
    
    def __len__(self):
        return len(self.patch_coords)
    
    def __getitem__(self, idx):
        """获取一个 patch"""
        d, h, w = self.patch_coords[idx]
        pd, ph, pw = self.patch_size
        
        # 提取 patch
        patch = self.volume[d:d+pd, h:h+ph, w:w+pw]
        
        # 归一化
        patch = self._normalize(patch)
        
        # 转换为张量
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)  # (1, D, H, W)
        
        return patch_tensor, (d, h, w)


def test_dataset():
    """测试数据集"""
    print("测试 Vesuvius 数据集...")
    
    # 创建测试数据
    data_dir = Path("data/processed/train")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建示例数据
    test_volume = np.random.rand(128, 128, 128).astype(np.float32)
    test_mask = (np.random.rand(128, 128, 128) > 0.5).astype(np.float32)
    
    np.save(data_dir / "volume.npy", test_volume)
    np.save(data_dir / "mask.npy", test_mask)
    
    print("\n1. 测试训练数据集")
    dataset = VesuviusDataset(
        data_dir=data_dir,
        patch_size=(64, 64, 64),
        augment=True
    )
    
    print(f"   数据集大小: {len(dataset)}")
    
    if len(dataset) > 0:
        volume, mask = dataset[0]
        print(f"   Volume shape: {volume.shape}")
        print(f"   Mask shape: {mask.shape}")
        print(f"   Volume range: [{volume.min():.4f}, {volume.max():.4f}]")
        print(f"   Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
    
    print("\n2. 测试推理数据集")
    inference_dataset = VesuviusInferenceDataset(
        volume_path=data_dir / "volume.npy",
        patch_size=(64, 64, 64),
        overlap=0.5
    )
    
    print(f"   数据集大小: {len(inference_dataset)}")
    
    if len(inference_dataset) > 0:
        patch, coords = inference_dataset[0]
        print(f"   Patch shape: {patch.shape}")
        print(f"   Coordinates: {coords}")
    
    print("\n✓ 数据集测试通过")


if __name__ == "__main__":
    test_dataset()
