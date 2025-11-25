# Vesuvius Challenge - 快速启动指南

**目标**: 在 AutoDL RTX 5090 上快速开始训练

---

## 🚀 5 分钟快速启动

### Step 1: 租用 AutoDL 实例

1. 访问 https://www.autodl.com/market/list
2. 选择 **RTX 5090** (24GB)
3. 配置:
   - GPU: RTX 5090 × 1
   - CPU: 16核
   - 内存: 64GB
   - 存储: 500GB SSD
4. 点击"租用"

**预计费用**: ~2.5元/小时

### Step 2: 环境配置

```bash
# SSH 登录到 AutoDL 实例
ssh root@your-instance-ip

# 克隆代码
git clone https://github.com/YOUR_USERNAME/vesuvius-challenge.git
cd vesuvius-challenge

# 安装依赖
pip install -r requirements.txt

# 配置 Kaggle API
mkdir -p ~/.kaggle
# 上传 kaggle.json 到 ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: 下载数据

```bash
# 下载比赛数据
python download_data.py

# 预处理数据（可选，在 Kaggle Notebook 完成更省钱）
python preprocess.py
```

### Step 4: 开始训练

```bash
# 使用基线配置训练
python train.py --config configs/baseline.yaml

# 或使用轻量级模型（更省显存）
python train.py --config configs/lite.yaml
```

### Step 5: 监控训练

```bash
# 查看 WandB 仪表板
# 访问 https://wandb.ai/your-username/vesuvius-challenge

# 或查看本地日志
tail -f logs/training.log
```

---

## 💰 成本优化技巧

### 1. 数据预处理在 Kaggle 完成

**在 Kaggle Notebook 运行**（免费 30h/week GPU）:

```python
# preprocess_kaggle.ipynb
import zarr
import numpy as np
from pathlib import Path

# 加载数据
volume = zarr.open('/kaggle/input/vesuvius-challenge-surface-detection/train/volume.zarr')

# 预处理
volume_norm = (volume - volume.mean()) / volume.std()

# 保存为 npy
np.save('volume_processed.npy', volume_norm)

# 下载到本地，然后上传到 AutoDL
```

### 2. 仅在训练时租用 GPU

```bash
# 训练前：租用 GPU
# 训练中：保持运行
# 训练后：立即释放

# 使用检查点恢复训练
python train.py --resume models/checkpoints/checkpoint_epoch_10.pth
```

### 3. 使用更小的模型

```yaml
# configs/lite.yaml
model:
  type: 'unet3d_lite'  # 轻量级模型
  base_channels: 16    # 更少的通道数
```

### 4. 减小 Patch Size

```yaml
data:
  patch_size: [48, 48, 48]  # 从 64 减小到 48
```

---

## 📊 训练监控

### WandB 配置

```bash
# 登录 WandB
wandb login

# 在 train.py 中已集成
# 访问 https://wandb.ai 查看实时训练曲线
```

### 关键指标

- **Train Loss**: 训练损失
- **Val Loss**: 验证损失
- **Dice Score**: Dice 系数（越高越好）
- **IoU**: 交并比
- **Learning Rate**: 学习率变化

---

## 🎯 训练策略

### 阶段 1: 快速验证（1-2 小时）

```yaml
training:
  epochs: 10
  batch_size: 2
```

**目标**: 验证代码可运行，模型可收敛

### 阶段 2: 基线训练（10-20 小时）

```yaml
training:
  epochs: 50
  batch_size: 2
  accumulation_steps: 4
```

**目标**: 获得基线分数

### 阶段 3: 优化训练（20-40 小时）

- 数据增强
- 超参数调优
- 模型集成

**目标**: 提升分数，冲击 Top 10%

---

## 🔧 常见问题

### Q1: 显存不足 (OOM)

**解决方案**:
```yaml
# 减小 batch size
batch_size: 1

# 减小 patch size
patch_size: [48, 48, 48]

# 使用轻量级模型
model:
  type: 'unet3d_lite'
```

### Q2: 训练速度慢

**解决方案**:
```yaml
# 增加 num_workers
num_workers: 8

# 使用混合精度（已默认开启）
mixed_precision: true

# 减少数据增强
augmentation:
  random_flip: true
  # 关闭其他增强
```

### Q3: 模型不收敛

**解决方案**:
```yaml
# 降低学习率
learning_rate: 0.00005

# 增加 warmup
scheduler:
  type: 'cosine_warmup'
  warmup_epochs: 5

# 检查数据预处理
```

### Q4: 如何恢复训练

```bash
# 从检查点恢复
python train.py --resume models/checkpoints/checkpoint_epoch_20.pth
```

---

## 📈 提交流程

### 1. 生成预测

```bash
python inference.py \
  --checkpoint models/checkpoints/best_model.pth \
  --test_dir data/test \
  --output submissions/submission.csv
```

### 2. 提交到 Kaggle

```bash
# 使用 Kaggle API
kaggle competitions submit \
  -c vesuvius-challenge-surface-detection \
  -f submissions/submission.csv \
  -m "Baseline submission"
```

### 3. 查看结果

访问 https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/submissions

---

## 🎯 优化路线图

### Week 1: 基线
- ✅ 环境配置
- ✅ 数据下载
- ✅ 基线训练
- ✅ 首次提交

### Week 2: 优化
- 🔄 数据增强优化
- 🔄 超参数调优
- 🔄 模型架构改进

### Week 3: 进阶
- 🔄 多模型集成
- 🔄 伪标签
- 🔄 后处理优化

### Week 4: 冲刺
- 🔄 最终优化
- 🔄 多次提交
- 🔄 冲击 Top 10%

---

## 📞 获取帮助

遇到问题？

1. 查看 [COMPETITION_PLAN.md](COMPETITION_PLAN.md)
2. 查看 Kaggle Discussion
3. 联系我获取支持

---

**祝您比赛顺利！** 🏆
