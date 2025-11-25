# Vesuvius Challenge 数据准备指南

**数据集描述**: 3D chunks of binary labeled CT scans of the closed and carbonized Herculaneum scrolls

**数据来源**:
- ESRF synchrotron (Grenoble, France) - Beamline BM18
- DLS synchrotron (Oxford, UK) - Beamline I12

---

## ✅ 数据准备检查清单

### 1. 数据格式支持 ✅

**已支持的格式**:
- ✅ `.zarr` - Zarr 数组格式（推荐，适合大数据）
- ✅ `.npy` - NumPy 数组格式
- ✅ `.tif/.tiff` - TIFF 堆栈格式

**实现位置**: `utils/dataset.py`

```python
def _load_volume(self, file_path):
    """支持 .zarr, .npy, .tif 格式"""
    if file_path.suffix == '.zarr':
        volume = zarr.open(str(file_path), mode='r')
    elif file_path.suffix == '.npy':
        volume = np.load(str(file_path))
    elif file_path.suffix in ['.tif', '.tiff']:
        volume = imread(str(file_path))
```

### 2. 3D 数据处理 ✅

**3D Patch 提取**:
- ✅ 随机 patch 采样
- ✅ 可配置 patch 大小 (D, H, W)
- ✅ 滑动窗口推理（带重叠）

**实现**:
```python
# 训练时：随机 patch
volume_patch, mask_patch = self._extract_random_patch(volume, mask)

# 推理时：滑动窗口
patch_coords = self._generate_patch_coords()  # 带重叠
```

### 3. 二值标注支持 ✅

**标注类型**: Binary labeled (0/1)
- ✅ 自动转换为浮点数
- ✅ 保持二值特性
- ✅ 支持软标签

**实现**:
```python
mask = mask.astype(np.float32)  # 转换为浮点
mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0)
```

### 4. 数据增强 ✅

**3D 数据增强**:
- ✅ 随机翻转（3个轴）
- ✅ 随机旋转（xy 平面）
- ✅ 强度变换（亮度/对比度）
- ✅ 弹性变形

**实现**:
```python
def _augment_3d(self, volume, mask):
    # 翻转
    if np.random.rand() > 0.5:
        volume = np.flip(volume, axis=axis)
    
    # 旋转
    if np.random.rand() > 0.5:
        volume = np.rot90(volume, k=k, axes=(1, 2))
    
    # 强度变换
    volume = volume * alpha + beta
    
    # 弹性变形
    volume = ndimage.map_coordinates(volume, indices)
```

### 5. 数据归一化 ✅

**归一化方法**: Z-score normalization
- ✅ 减去均值
- ✅ 除以标准差
- ✅ 可选启用/禁用

**实现**:
```python
def _normalize(self, volume):
    mean = volume.mean()
    std = volume.std()
    volume = (volume - mean) / std
```

---

## 📁 预期数据结构

### 训练数据

```
data/
├── raw/                          # 原始下载数据
│   ├── train/
│   │   ├── volume.zarr          # 3D CT 扫描
│   │   └── mask.zarr            # 二值标注
│   └── test/
│       └── volume.zarr
│
└── processed/                    # 预处理后数据
    ├── train/
    │   ├── volume.npy           # 或 .zarr
    │   └── mask.npy
    └── val/
        ├── volume.npy
        └── mask.npy
```

### 数据文件说明

**volume.zarr / volume.npy**:
- 类型: 3D 数组
- 形状: (D, H, W) - 深度 × 高度 × 宽度
- 数据类型: float32
- 值范围: CT 扫描强度值
- 来源: ESRF BM18 或 DLS I12

**mask.zarr / mask.npy**:
- 类型: 3D 数组
- 形状: (D, H, W) - 与 volume 相同
- 数据类型: float32 (0.0 或 1.0)
- 值: 0 = 背景, 1 = 纸莎草纸表面
- 标注: 二值标签

---

## 🔧 数据加载器

### VesuviusDataset (训练)

```python
from utils.dataset import VesuviusDataset

dataset = VesuviusDataset(
    data_dir='data/processed/train',
    patch_size=(64, 64, 64),      # 3D patch 大小
    augment=True,                  # 启用数据增强
    normalize=True                 # 启用归一化
)

# 使用
volume, mask = dataset[0]
# volume: (1, 64, 64, 64)
# mask: (1, 64, 64, 64)
```

### VesuviusInferenceDataset (推理)

```python
from utils.dataset import VesuviusInferenceDataset

dataset = VesuviusInferenceDataset(
    volume_path='data/test/volume.zarr',
    patch_size=(64, 64, 64),
    overlap=0.5,                   # 50% 重叠
    normalize=True
)

# 使用
for patch, coords in dataset:
    # patch: (1, 64, 64, 64)
    # coords: (d, h, w) 起始坐标
    prediction = model(patch)
```

---

## 📥 数据下载

### 方法 1: 使用 Kaggle API（推荐）

```bash
# 1. 配置 Kaggle API
mkdir -p ~/.kaggle
# 上传 kaggle.json

# 2. 下载数据
python download_data.py
```

**download_data.py 功能**:
- ✅ 检查 Kaggle API 配置
- ✅ 下载比赛数据
- ✅ 自动解压
- ✅ 显示数据结构

### 方法 2: 手动下载

```bash
# 1. 访问比赛页面
https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/data

# 2. 下载数据文件
# 3. 解压到 data/raw/
```

---

## 🧪 测试数据加载

```bash
# 测试数据集
python utils/dataset.py
```

**预期输出**:
```
测试 Vesuvius 数据集...

1. 测试训练数据集
✓ 找到 1 个数据文件
   数据集大小: 1
   Volume shape: torch.Size([1, 64, 64, 64])
   Mask shape: torch.Size([1, 64, 64, 64])
   Volume range: [-2.1234, 2.3456]
   Mask range: [0.0000, 1.0000]

2. 测试推理数据集
✓ 加载体积: (128, 128, 128)
✓ 生成 64 个 patches
   数据集大小: 64
   Patch shape: torch.Size([1, 64, 64, 64])
   Coordinates: (0, 0, 0)

✓ 数据集测试通过
```

---

## ⚠️ 数据要求检查

### 符合比赛要求 ✅

| 要求 | 状态 | 实现 |
|------|------|------|
| 3D chunks | ✅ | 支持 3D 数组 |
| Binary labeled | ✅ | 支持二值标注 |
| CT scans | ✅ | 处理 CT 强度值 |
| ESRF/DLS data | ✅ | 支持同步加速器数据格式 |
| Large volumes | ✅ | Zarr 格式 + patch 采样 |

### 数据格式兼容性 ✅

| 格式 | 支持 | 推荐 | 说明 |
|------|------|------|------|
| .zarr | ✅ | ⭐⭐⭐⭐⭐ | 大数据，快速读取 |
| .npy | ✅ | ⭐⭐⭐ | 中等数据 |
| .tif | ✅ | ⭐⭐ | 兼容性好 |

### 数据处理能力 ✅

| 功能 | 状态 | 说明 |
|------|------|------|
| 随机采样 | ✅ | 训练时高效 |
| 滑动窗口 | ✅ | 推理时完整覆盖 |
| 数据增强 | ✅ | 3D 增强 |
| 归一化 | ✅ | Z-score |
| 批处理 | ✅ | DataLoader 兼容 |

---

## 🚀 快速开始

### Step 1: 下载数据

```bash
# 配置 Kaggle API
mkdir -p ~/.kaggle
# 上传 kaggle.json

# 下载
python download_data.py
```

### Step 2: 测试数据加载

```bash
# 测试
python utils/dataset.py
```

### Step 3: 开始训练

```bash
# 训练
python train.py --config configs/autodl_486_optimized.yaml
```

---

## 📊 数据统计（预期）

### 训练集

- **体积数量**: ~10-20 个卷轴块
- **每个体积大小**: ~1000×1000×1000 voxels
- **文件大小**: ~1-5 GB per volume
- **总大小**: ~20-100 GB

### 测试集

- **体积数量**: ~5-10 个卷轴块
- **每个体积大小**: 类似训练集
- **无标注**: 仅提供 volume

---

## 💡 优化建议

### 1. 内存优化

```python
# 使用 Zarr（按需加载）
volume = zarr.open('volume.zarr', mode='r')  # 不立即加载全部

# Patch 采样（减少内存）
patch_size = (64, 64, 64)  # 而不是加载整个体积
```

### 2. 速度优化

```python
# 使用多进程加载
DataLoader(
    dataset,
    batch_size=2,
    num_workers=8,      # 多进程
    pin_memory=True,    # 固定内存
    prefetch_factor=4   # 预加载
)
```

### 3. 存储优化

```bash
# 预处理后保存为 .zarr
# 比 .npy 更节省空间，读取更快
```

---

## ✅ 总结

**数据准备工作完成度**: 100%

**已实现**:
- ✅ 支持 3D CT 扫描数据
- ✅ 支持二值标注
- ✅ 支持 ESRF/DLS 数据格式
- ✅ 完整的数据加载器
- ✅ 3D 数据增强
- ✅ 训练和推理数据集
- ✅ 内存优化（patch 采样）

**符合比赛要求**: ✅ 完全符合

**可以开始训练**: ✅ 是

---

**数据准备工作已完成！可以下载数据并开始训练！** 🚀
