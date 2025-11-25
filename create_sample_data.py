"""
创建测试数据

生成小规模的 3D CT 扫描数据用于快速测试
"""

import numpy as np
from pathlib import Path
from scipy import ndimage


def create_synthetic_scroll(size=(128, 128, 128), num_layers=3):
    """
    创建合成的卷轴 CT 扫描数据
    
    模拟纸莎草纸卷轴的 3D 结构
    
    Parameters
    ----------
    size : tuple
        体积大小 (D, H, W)
    num_layers : int
        纸莎草纸层数
    
    Returns
    -------
    tuple
        (volume, mask) - CT 扫描和二值标注
    """
    D, H, W = size
    
    # 初始化
    volume = np.random.randn(*size).astype(np.float32) * 0.1  # 背景噪声
    mask = np.zeros(size, dtype=np.float32)
    
    # 创建卷轴中心
    center_z = D // 2
    center_y = H // 2
    center_x = W // 2
    
    # 为每一层创建螺旋状的纸莎草纸
    for layer_idx in range(num_layers):
        radius = 20 + layer_idx * 8  # 每层半径递增
        thickness = 2  # 纸张厚度
        
        # 创建圆柱形的层
        for z in range(D):
            for y in range(H):
                for x in range(W):
                    # 计算到中心的距离
                    dy = y - center_y
                    dx = x - center_x
                    dist = np.sqrt(dy**2 + dx**2)
                    
                    # 如果在这一层的半径范围内
                    if radius - thickness < dist < radius + thickness:
                        # 添加螺旋扭曲
                        angle = np.arctan2(dy, dx)
                        z_offset = int(angle * 5)  # 螺旋效果
                        
                        if abs(z - center_z - z_offset) < 10:
                            # CT 扫描强度（纸莎草纸比背景亮）
                            volume[z, y, x] = 1.0 + np.random.randn() * 0.2
                            
                            # 标注（表面）
                            mask[z, y, x] = 1.0
    
    # 添加一些噪声和伪影
    volume += np.random.randn(*size) * 0.1
    
    # 平滑处理（模拟真实 CT 扫描）
    volume = ndimage.gaussian_filter(volume, sigma=1.0)
    
    # 归一化到 [0, 1]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    
    return volume, mask


def create_train_val_split(train_size=(128, 128, 128), val_size=(128, 128, 128)):
    """
    创建训练集和验证集
    
    Parameters
    ----------
    train_size : tuple
        训练数据大小
    val_size : tuple
        验证数据大小
    """
    print("=" * 60)
    print("创建测试数据")
    print("=" * 60)
    print()
    
    # 创建目录
    data_dir = Path("data/processed")
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建训练数据
    print("📦 创建训练数据...")
    train_volume, train_mask = create_synthetic_scroll(size=train_size, num_layers=3)
    
    np.save(train_dir / "volume.npy", train_volume)
    np.save(train_dir / "mask.npy", train_mask)
    
    print(f"   ✓ 训练体积: {train_volume.shape}")
    print(f"   ✓ 训练掩码: {train_mask.shape}")
    print(f"   ✓ 掩码覆盖率: {train_mask.mean() * 100:.2f}%")
    
    # 创建验证数据
    print("\n📦 创建验证数据...")
    val_volume, val_mask = create_synthetic_scroll(size=val_size, num_layers=2)
    
    np.save(val_dir / "volume.npy", val_volume)
    np.save(val_dir / "mask.npy", val_mask)
    
    print(f"   ✓ 验证体积: {val_volume.shape}")
    print(f"   ✓ 验证掩码: {val_mask.shape}")
    print(f"   ✓ 掩码覆盖率: {val_mask.mean() * 100:.2f}%")
    
    # 显示统计信息
    print("\n📊 数据统计:")
    print(f"   训练集:")
    print(f"     - Volume range: [{train_volume.min():.4f}, {train_volume.max():.4f}]")
    print(f"     - Mask pixels: {train_mask.sum():.0f} / {train_mask.size}")
    print(f"   验证集:")
    print(f"     - Volume range: [{val_volume.min():.4f}, {val_volume.max():.4f}]")
    print(f"     - Mask pixels: {val_mask.sum():.0f} / {val_mask.size}")
    
    print("\n" + "=" * 60)
    print("✅ 测试数据创建完成！")
    print("=" * 60)
    print("\n下一步:")
    print("  python train.py --config configs/test.yaml")


def create_test_data():
    """创建测试数据（用于推理）"""
    print("\n📦 创建测试数据...")
    
    test_dir = Path("data/processed/test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建测试体积（无标注）
    test_volume, _ = create_synthetic_scroll(size=(128, 128, 128), num_layers=3)
    
    np.save(test_dir / "volume.npy", test_volume)
    
    print(f"   ✓ 测试体积: {test_volume.shape}")
    print(f"   ✓ 保存路径: {test_dir / 'volume.npy'}")


if __name__ == "__main__":
    # 创建训练和验证数据
    create_train_val_split(
        train_size=(128, 128, 128),
        val_size=(128, 128, 128)
    )
    
    # 创建测试数据
    create_test_data()
