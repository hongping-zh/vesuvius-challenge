# 🚀 优化实施路线图

**基于完整测试结果的优化建议**

---

## 📊 当前状态

### 已实现 ✅
- ✅ DynUNet (365M 参数)
- ✅ Deep Supervision
- ✅ Vesuvius 组合损失
- ✅ 拓扑后处理
- ✅ 所有测试通过

### 预期性能
- 基础 DynUNet: **0.65-0.70**
- 优化后: **0.75-0.80**
- 提升空间: **+0.10~0.15**

---

## 🎯 三阶段优化策略

### 阶段 1: 验证基线（今天，4-5h）

**目标**: 验证 DynUNet 基础性能

**步骤**:
```bash
# 在 AutoDL 上
python train.py --config configs/autodl_dynunet_small.yaml
```

**成功标准**:
- ✅ SurfaceDice > 0.65
- ✅ 训练稳定
- ✅ 无错误

**如果成功**: → 阶段 2  
**如果失败**: → 调试

---

### 阶段 2: 核心优化（明天，10-12h）

**目标**: 实现关键优化，提升 +0.10~0.15

#### 优化 1: Ink-only Sampling ⭐⭐⭐⭐⭐

**问题**: 墨迹极不平衡（<0.1% 正像素）

**实现**:

```python
# utils/dataset.py 中添加

class InkAwareVesuviusDataset(VesuviusDataset):
    """墨迹感知数据集"""
    
    def __init__(
        self,
        data_dir,
        patch_size,
        positive_ratio=0.7,
        min_ink_pixels=100,
        **kwargs
    ):
        super().__init__(data_dir, patch_size, **kwargs)
        self.positive_ratio = positive_ratio
        self.min_ink_pixels = min_ink_pixels
        
        # 预先扫描哪些 patch 包含墨迹
        self._build_ink_index()
    
    def _build_ink_index(self):
        """构建墨迹索引"""
        print("🔍 扫描墨迹分布...")
        self.ink_patches = []
        self.no_ink_patches = []
        
        # 这里需要扫描整个 volume
        # 记录哪些位置有墨迹
        # 实现细节见下方
        
    def __getitem__(self, idx):
        """采样时优先选择包含墨迹的 patch"""
        if np.random.rand() < self.positive_ratio:
            # 采样包含墨迹的 patch
            patch_idx = np.random.choice(self.ink_patches)
        else:
            # 采样背景 patch
            patch_idx = np.random.choice(self.no_ink_patches)
        
        return self._extract_patch(patch_idx)
```

**配置**:
```yaml
data:
  dataset_type: 'ink_aware'
  positive_ratio: 0.7
  min_ink_pixels: 100
```

**预期提升**: +0.10~0.15  
**实现时间**: 4-6 小时

---

#### 优化 2: 多通道输入 ⭐⭐⭐⭐

**问题**: 只用 raw intensity，信息不足

**实现**:

```python
# utils/features.py (新建)

import numpy as np
from scipy import ndimage

def compute_gradient_features(volume):
    """计算梯度特征"""
    grad_x = np.gradient(volume, axis=0)
    grad_y = np.gradient(volume, axis=1)
    grad_z = np.gradient(volume, axis=2)
    return grad_x, grad_y, grad_z

def compute_log_features(volume, sigma=1.0):
    """计算 LoG 特征"""
    log = ndimage.gaussian_laplace(volume, sigma=sigma)
    return log

def compute_hessian_features(volume):
    """计算 Hessian 特征"""
    # 简化版：只计算主要特征值
    grad_x = np.gradient(volume, axis=0)
    grad_y = np.gradient(volume, axis=1)
    grad_z = np.gradient(volume, axis=2)
    
    # 二阶导数
    hxx = np.gradient(grad_x, axis=0)
    hyy = np.gradient(grad_y, axis=1)
    hzz = np.gradient(grad_z, axis=2)
    
    return hxx, hyy, hzz

def extract_multi_channel_features(volume, channels=['raw', 'grad', 'log']):
    """提取多通道特征"""
    features = []
    
    if 'raw' in channels:
        features.append(volume)
    
    if 'grad' in channels:
        grad_x, grad_y, grad_z = compute_gradient_features(volume)
        features.extend([grad_x, grad_y, grad_z])
    
    if 'log' in channels:
        log = compute_log_features(volume)
        features.append(log)
    
    if 'hessian' in channels:
        hxx, hyy, hzz = compute_hessian_features(volume)
        features.extend([hxx, hyy, hzz])
    
    # Stack to (C, D, H, W)
    return np.stack(features, axis=0)
```

**修改 Dataset**:
```python
# utils/dataset.py

class MultiChannelVesuviusDataset(VesuviusDataset):
    def __init__(self, data_dir, patch_size, channels=['raw', 'grad'], **kwargs):
        super().__init__(data_dir, patch_size, **kwargs)
        self.channels = channels
    
    def __getitem__(self, idx):
        # 加载 volume
        volume = self._load_volume()
        
        # 提取多通道特征
        features = extract_multi_channel_features(volume, self.channels)
        
        # 提取 patch
        patch = self._extract_patch(features, idx)
        
        return patch, mask
```

**配置**:
```yaml
model:
  in_channels: 4  # raw + grad_x + grad_y + grad_z

data:
  channels: ['raw', 'grad']
```

**预期提升**: +0.05~0.10  
**实现时间**: 2-3 小时

---

#### 优化 3: 动态 Loss 权重 ⭐⭐⭐⭐

**问题**: 固定权重不够灵活

**实现**:

```python
# utils/losses.py 中添加

class DynamicLossScheduler:
    """动态损失权重调度器"""
    
    def __init__(self, total_epochs, warmup_epochs=20):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
    
    def get_weights(self, epoch):
        """获取当前 epoch 的权重"""
        if epoch < self.warmup_epochs:
            # 前期：只学习基础分割
            return {
                'dice': 0.5,
                'bce': 0.5,
                'surface': 0.0,
                'centerline': 0.0,
                'topology': 0.0
            }
        else:
            # 后期：加入拓扑约束
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            surface_weight = 0.2 * progress
            topology_weight = 0.2 * progress
            
            return {
                'dice': 0.5 - 0.1 * progress,
                'bce': 0.5 - 0.1 * progress,
                'surface': surface_weight,
                'centerline': 0.0,
                'topology': topology_weight
            }
```

**修改训练循环**:
```python
# train.py

# 创建调度器
loss_scheduler = DynamicLossScheduler(
    total_epochs=config['training']['epochs'],
    warmup_epochs=20
)

# 在每个 epoch 开始时
for epoch in range(epochs):
    # 更新损失权重
    weights = loss_scheduler.get_weights(epoch)
    criterion.update_weights(weights)
    
    # 训练...
```

**预期提升**: +0.05~0.08  
**实现时间**: 2-3 小时

---

### 阶段 3: 完整训练（3-4天，50h）

**目标**: 冲击 Top 10%

**步骤**:
1. 使用优化后的配置
2. 训练 50 epochs
3. 保存最佳模型
4. Kaggle 提交

**配置**:
```yaml
model:
  type: dynunet
  in_channels: 4  # 多通道
  base_num_features: 64
  deep_supervision: true

data:
  dataset_type: 'ink_aware'  # Ink-only sampling
  channels: ['raw', 'grad']
  positive_ratio: 0.7
  patch_size: [128, 128, 128]  # 更大 patch

training:
  epochs: 50
  batch_size: 1
  accumulation_steps: 8
  use_dynamic_loss: true  # 动态权重
```

**预期结果**: SurfaceDice 0.75-0.80

---

## 💰 成本效益分析

### 方案对比

| 方案 | 优化项 | 开发时间 | 训练时间 | 成本 | 预期分数 |
|------|--------|----------|----------|------|----------|
| A | 仅 DynUNet | 0h | 50h | 150元 | 0.70 |
| B | + Ink-only | 6h | 50h | 150元 | 0.73 |
| C | + Ink-only + 多通道 | 10h | 50h | 150元 | 0.75 |
| D | + 所有 P1 优化 | 14h | 50h | 150元 | 0.78 |

**推荐**: 方案 C（性价比最高）

---

## 📋 实施检查清单

### 阶段 1: 验证基线

- [ ] 上传代码到 AutoDL
- [ ] 安装依赖
- [ ] 下载数据
- [ ] 开始训练（8 epochs）
- [ ] 验证 SurfaceDice > 0.65

### 阶段 2: 核心优化

- [ ] 实现 Ink-only Sampling
- [ ] 测试 Ink-only Sampling
- [ ] 实现多通道输入
- [ ] 测试多通道输入
- [ ] 实现动态 Loss 权重
- [ ] 验证优化（8 epochs）
- [ ] 确认提升 > +0.10

### 阶段 3: 完整训练

- [ ] 更新配置（50 epochs）
- [ ] 开始完整训练
- [ ] 监控训练进度
- [ ] 保存最佳模型
- [ ] Kaggle 提交
- [ ] 查看 Public Score

---

## 🎯 决策树

```
开始
  ↓
验证 DynUNet (8 epochs)
  ↓
SurfaceDice > 0.65?
  ├─ 是 → 实施核心优化
  │        ↓
  │      验证优化 (8 epochs)
  │        ↓
  │      提升 > +0.10?
  │        ├─ 是 → 完整训练 (50 epochs)
  │        │        ↓
  │        │      Final Score > 0.75?
  │        │        ├─ 是 → 🎉 成功！
  │        │        └─ 否 → Ensemble
  │        └─ 否 → 调试优化
  │
  └─ 否 → 调试配置
           ↓
         重新验证
```

---

## 📞 快速参考

### 关键文件

```
优化相关:
├── utils/features.py          # 多通道特征（待创建）
├── utils/dataset.py           # Ink-only sampling（待修改）
├── utils/losses.py            # 动态权重（待修改）
└── configs/autodl_optimized.yaml  # 优化配置（待创建）
```

### 关键命令

```bash
# 阶段 1: 验证基线
python train.py --config configs/autodl_dynunet_small.yaml

# 阶段 2: 验证优化
python train.py --config configs/autodl_optimized.yaml --epochs 8

# 阶段 3: 完整训练
python train.py --config configs/autodl_optimized.yaml --epochs 50
```

---

## 🎊 总结

### 推荐路线

**今天**: 验证 DynUNet 基线  
**明天**: 实施核心优化（如果基线成功）  
**后天**: 开始完整训练（如果优化有效）

### 预期成果

**保守**: 0.70 (仅 DynUNet)  
**目标**: 0.75 (DynUNet + 核心优化)  
**理想**: 0.78 (DynUNet + 所有优化)

### 风险控制

**低风险**: 先验证，再优化  
**中风险**: 分阶段实施  
**高风险**: 每步都有回退方案

---

**🚀 准备好了！先验证基线，再决定是否优化！**
