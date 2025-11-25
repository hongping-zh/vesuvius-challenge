# Vesuvius Challenge - Surface Detection 参赛方案

**比赛链接**: https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection  
**计算资源**: AutoDL 5090 RTX  
**预算**: 有限（需要优化成本）

---

## 🎯 比赛概述

### 任务
虚拟展开古代赫库兰尼姆卷轴，检测表面和墨迹

### 核心挑战
1. **表面检测** - 从 3D CT 扫描中识别纸莎草纸表面
2. **墨迹识别** - 在虚拟展开的表面上检测碳墨迹
3. **大数据处理** - CT 扫描数据量巨大（TB 级）

---

## 💰 资源优化策略

### AutoDL 5090 RTX 配置建议

**推荐配置**:
- GPU: RTX 5090 (24GB VRAM)
- CPU: 16核+
- 内存: 64GB+
- 存储: 500GB+ SSD

**成本优化**:
1. **按需租用** - 仅在训练时租用
2. **数据预处理** - 本地或 Kaggle Notebook 完成
3. **混合精度训练** - 使用 FP16/BF16 减少显存
4. **梯度累积** - 小 batch size + 梯度累积
5. **检查点保存** - 随时暂停恢复

---

## 📊 技术方案

### 方案 A: 3D U-Net（推荐）

**优势**:
- 直接处理 3D CT 数据
- 适合表面检测
- 成熟的医学图像分割架构

**架构**:
```
Input: 3D CT Volume (D×H×W)
↓
3D U-Net Encoder (5 levels)
↓
Bottleneck
↓
3D U-Net Decoder (5 levels)
↓
Output: Surface Mask (D×H×W)
```

**显存优化**:
- Patch-based training (64×64×64)
- Mixed precision (FP16)
- Gradient checkpointing
- 预计显存: 16-20GB

### 方案 B: 2.5D 方法（备选）

**优势**:
- 显存需求更小
- 训练速度更快
- 适合资源受限

**方法**:
- 将 3D 数据切片为 2D + depth channel
- 使用 2D U-Net/ResNet
- 多切片融合

---

## 🚀 实施计划

### Phase 1: 环境准备（Day 1）

1. **AutoDL 环境配置**
```bash
# 创建环境
conda create -n vesuvius python=3.10
conda activate vesuvius

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch
pip install albumentations
pip install opencv-python
pip install zarr
pip install tifffile
pip install wandb
```

2. **数据下载**
```bash
# 使用 Kaggle API
pip install kaggle
kaggle competitions download -c vesuvius-challenge-surface-detection
```

### Phase 2: 数据预处理（Day 1-2）

**在 Kaggle Notebook 完成（免费 GPU）**:

```python
# 数据加载和预处理
import zarr
import numpy as np

def preprocess_volume(volume_path, output_path):
    # 加载 zarr/tiff 数据
    volume = zarr.open(volume_path, mode='r')
    
    # 归一化
    volume_norm = (volume - volume.mean()) / volume.std()
    
    # 保存为 npy（便于快速加载）
    np.save(output_path, volume_norm)
```

**数据增强策略**:
- Random crop (64×64×64)
- Random flip (x, y, z)
- Random rotation (±15°)
- Elastic deformation
- Intensity shift

### Phase 3: 模型训练（Day 3-5）

**基线模型 - 3D U-Net**:

```python
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # 使用 smp 的 2D U-Net 作为基础
        # 扩展为 3D
        pass
    
    def forward(self, x):
        return x

# 训练配置
config = {
    'batch_size': 2,  # 受限于显存
    'patch_size': (64, 64, 64),
    'learning_rate': 1e-4,
    'epochs': 50,
    'mixed_precision': True,
    'gradient_accumulation': 4,  # 等效 batch_size=8
}
```

**损失函数**:
```python
# 组合损失
loss = 0.5 * DiceLoss() + 0.5 * BCEWithLogitsLoss()
```

**优化器**:
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50
)
```

### Phase 4: 推理和提交（Day 6-7）

**推理策略**:
- Sliding window (overlap=0.5)
- Test-time augmentation (TTA)
- 多模型集成

**提交格式**:
```python
# 生成提交文件
submission = pd.DataFrame({
    'id': test_ids,
    'prediction': predictions
})
submission.to_csv('submission.csv', index=False)
```

---

## 💡 优化技巧

### 1. 显存优化

```python
# 混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. 梯度累积

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 检查点保存

```python
# 每 epoch 保存
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, f'checkpoint_epoch_{epoch}.pth')
```

### 4. 数据加载优化

```python
# 使用 DataLoader 的多进程
dataloader = DataLoader(
    dataset,
    batch_size=2,
    num_workers=4,  # 多进程加载
    pin_memory=True,  # 加速 GPU 传输
    prefetch_factor=2
)
```

---

## 📈 训练监控

### WandB 集成

```python
import wandb

wandb.init(
    project="vesuvius-challenge",
    config=config
)

# 记录指标
wandb.log({
    'train_loss': train_loss,
    'val_loss': val_loss,
    'dice_score': dice_score
})
```

---

## 🎯 成功策略

### 短期目标（Week 1）
1. ✅ 环境配置完成
2. ✅ 数据下载和预处理
3. ✅ 基线模型训练
4. ✅ 首次提交

### 中期目标（Week 2-3）
1. 模型优化（更深的网络）
2. 数据增强优化
3. 超参数调优
4. 多模型集成

### 长期目标（Week 4+）
1. 高级技术（Attention, Transformer）
2. 伪标签（Semi-supervised）
3. 后处理优化
4. 冲击 Top 10%

---

## 💰 成本估算

### AutoDL 费用（RTX 5090）

**假设**: 2.5元/小时

**训练阶段**:
- 数据预处理: 2小时 × 2.5元 = 5元
- 基线训练: 20小时 × 2.5元 = 50元
- 优化训练: 30小时 × 2.5元 = 75元
- 推理: 5小时 × 2.5元 = 12.5元

**总计**: ~150元（一个月）

**省钱技巧**:
1. 使用 Kaggle Notebook（免费 30h/week GPU）
2. 本地预处理数据
3. 仅在训练时租用 GPU
4. 使用更小的模型（2.5D）

---

## 📚 参考资源

### 论文
1. 3D U-Net: https://arxiv.org/abs/1606.06650
2. nnU-Net: https://arxiv.org/abs/1809.10486
3. Vesuvius Challenge 获奖方案

### 代码
1. Segmentation Models PyTorch
2. MONAI (医学图像)
3. Kaggle Notebooks (公开方案)

### 数据
1. Vesuvius Challenge 官网
2. Kaggle 数据集
3. 训练样本和标注

---

## ⚠️ 风险和应对

### 风险 1: 显存不足
**应对**: 
- 减小 patch size
- 使用 2.5D 方法
- 梯度检查点

### 风险 2: 训练时间过长
**应对**:
- 使用预训练模型
- 减少 epoch 数
- Early stopping

### 风险 3: 过拟合
**应对**:
- 数据增强
- Dropout
- 正则化

---

## 🚀 快速启动

### 1. 克隆代码仓库
```bash
git clone https://github.com/YOUR_REPO/vesuvius-challenge.git
cd vesuvius-challenge
```

### 2. 配置环境
```bash
bash setup.sh
```

### 3. 下载数据
```bash
python download_data.py
```

### 4. 训练模型
```bash
python train.py --config configs/baseline.yaml
```

### 5. 生成提交
```bash
python inference.py --checkpoint best_model.pth
```

---

## 📞 支持

有问题随时联系！我会帮您：
1. 调试代码
2. 优化模型
3. 解决技术问题
4. 提供建议

---

**祝您比赛顺利！冲击 Top 10%！** 🏆
