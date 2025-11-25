# AutoDL 486机 RTX 5090 配置指南

**租用配置**: ✅ 已确认

---

## 🖥️ 您的配置

```
主机: 486机
GPU: RTX 5090 (32GB) - 1/8 卡
CPU: 25核 Xeon(R) Platinum 8470Q
内存: 90GB
数据盘: 50GB (可扩容至 5708GB)
驱动: 580.76.05
CUDA: 13.0

费用: ￥3.03/时
```

### 💰 成本分析

**优势**:
- ✅ RTX 5090 32GB（比计划的 24GB 多 8GB！）
- ✅ 90GB 内存（比计划的 64GB 多 26GB！）
- ✅ 25核 CPU（比计划的 16核多 9核！）
- ✅ 费用 3.03元/时（仅比预算高 0.53元/时）

**性价比**: ⭐⭐⭐⭐⭐ 极高！

**预算更新**:
- 快速验证: 2小时 × 3.03元 = 6.06元
- 基线训练: 20小时 × 3.03元 = 60.6元
- 优化训练: 30小时 × 3.03元 = 90.9元
- 推理测试: 5小时 × 3.03元 = 15.15元

**总计**: ~173元/月（比预算多 23元，但配置更好！）

---

## 🚀 立即开始配置

### Step 1: SSH 登录

```bash
# AutoDL 会提供 SSH 命令，类似：
ssh -p [端口] root@[IP地址]

# 例如：
ssh -p 12345 root@region-1.autodl.com
```

### Step 2: 检查环境

```bash
# 检查 GPU
nvidia-smi

# 应该看到：
# RTX 5090 32GB
# CUDA 13.0
# Driver 580.76.05

# 检查 CUDA
nvcc --version

# 检查 Python
python --version
```

### Step 3: 克隆代码

```bash
# 创建工作目录
mkdir -p /root/projects
cd /root/projects

# 克隆代码（如果已上传到 GitHub）
git clone https://github.com/YOUR_USERNAME/vesuvius-challenge.git
cd vesuvius-challenge

# 或者从本地上传
# 使用 scp 或 AutoDL 的文件上传功能
```

### Step 4: 创建 Conda 环境

```bash
# 创建环境
conda create -n vesuvius python=3.10 -y
conda activate vesuvius

# 安装 PyTorch (CUDA 13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

### Step 5: 配置 Kaggle API

```bash
# 创建 .kaggle 目录
mkdir -p ~/.kaggle

# 上传 kaggle.json
# 方法1: 使用 AutoDL 文件上传功能
# 方法2: 使用 scp
# scp kaggle.json root@[IP]:/root/.kaggle/

# 设置权限
chmod 600 ~/.kaggle/kaggle.json

# 测试
kaggle competitions list
```

---

## 📥 数据下载策略

### 方案 A: 直接下载到 AutoDL（推荐）

```bash
# 下载数据
python download_data.py

# 数据会保存到 data/raw/
```

**优势**: 
- 简单直接
- 数据在训练机器上

**注意**: 
- 数据盘只有 50GB
- 如果数据超过 50GB，需要扩容

### 方案 B: Kaggle Notebook 预处理（省钱）

**在 Kaggle Notebook 运行**:

```python
# 1. 加载数据
import zarr
import numpy as np

volume = zarr.open('/kaggle/input/vesuvius-challenge-surface-detection/train/volume.zarr')

# 2. 预处理
volume_norm = (volume - volume.mean()) / volume.std()

# 3. 保存为 npy
np.save('volume_processed.npy', volume_norm)

# 4. 下载到本地，然后上传到 AutoDL
```

**优势**:
- 节省 AutoDL 费用
- 利用 Kaggle 免费 GPU

---

## 🎯 优化配置建议

### 利用 32GB 显存优势

**原配置** (24GB):
```yaml
training:
  batch_size: 2
  patch_size: [64, 64, 64]
```

**新配置** (32GB) - 推荐：
```yaml
training:
  batch_size: 3  # 从 2 增加到 3
  patch_size: [80, 80, 80]  # 从 64 增加到 80
  accumulation_steps: 3  # 等效 batch_size = 9
```

**优势**:
- ✅ 更大的 batch size → 训练更稳定
- ✅ 更大的 patch size → 更多上下文信息
- ✅ 可能提升模型性能

### 利用 90GB 内存优势

```yaml
training:
  num_workers: 8  # 从 4 增加到 8
  prefetch_factor: 4  # 预加载更多数据
```

**优势**:
- ✅ 更快的数据加载
- ✅ 减少 GPU 等待时间
- ✅ 提升训练效率

### 利用 25核 CPU 优势

```python
# 数据预处理可以并行
import multiprocessing as mp

# 使用 20 个进程（留 5 个给系统）
pool = mp.Pool(processes=20)
```

---

## 📝 优化后的配置文件

创建新配置: `configs/autodl_486.yaml`

```yaml
# AutoDL 486机 RTX 5090 优化配置

model:
  type: 'unet3d'  # 使用标准版（显存充足）
  in_channels: 1
  out_channels: 1
  base_channels: 48  # 从 32 增加到 48

data:
  train_dir: 'data/processed/train'
  val_dir: 'data/processed/val'
  patch_size: [80, 80, 80]  # 增大 patch size
  
training:
  batch_size: 3  # 利用 32GB 显存
  accumulation_steps: 3  # 等效 batch_size = 9
  epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.00001
  num_workers: 8  # 利用 25核 CPU
  prefetch_factor: 4
  save_frequency: 5
  checkpoint_dir: 'models/checkpoints'

loss:
  type: 'dice_bce'
  dice_weight: 0.5
  bce_weight: 0.5

optimizer:
  type: 'adamw'
  betas: [0.9, 0.999]
  eps: 0.00000001

scheduler:
  type: 'cosine'
  T_max: 50

augmentation:
  random_flip: true
  random_rotation: 15
  elastic_deformation: true
  intensity_shift: 0.1

logging:
  use_wandb: true
  project: 'vesuvius-challenge'
  log_frequency: 10

inference:
  patch_size: [80, 80, 80]
  overlap: 0.5
  tta: true
  batch_size: 6  # 推理时可以更大
```

---

## 🔧 启动脚本

创建 `start_training.sh`:

```bash
#!/bin/bash

echo "=========================================="
echo "Vesuvius Challenge - AutoDL 486机"
echo "=========================================="
echo ""

# 激活环境
source activate vesuvius

# 检查 GPU
echo "检查 GPU..."
nvidia-smi

echo ""
echo "开始训练..."
echo ""

# 训练
python train.py --config configs/autodl_486.yaml

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
```

使用：
```bash
chmod +x start_training.sh
./start_training.sh
```

---

## 📊 预期性能

### 训练速度估算

**配置对比**:

| 配置 | Batch Size | Patch Size | 速度 (it/s) | Epoch 时间 |
|------|-----------|-----------|-------------|-----------|
| 原计划 (24GB) | 2 | 64³ | ~0.5 | 40分钟 |
| **486机 (32GB)** | **3** | **80³** | **~0.4** | **50分钟** |

**总训练时间**:
- 50 epochs × 50分钟 = ~42小时
- 费用: 42小时 × 3.03元 = **127元**

**优势**:
- ✅ 更大的模型容量
- ✅ 更好的性能
- ✅ 可能更高的分数

---

## 💾 数据盘管理

### 检查空间

```bash
# 查看磁盘使用
df -h

# 查看数据目录大小
du -sh data/
```

### 扩容建议

**如果数据超过 50GB**:

1. 在 AutoDL 控制台扩容数据盘
2. 建议扩容到 200-300GB
3. 费用: 按需计费

**优化存储**:
```bash
# 删除原始数据，只保留预处理后的
rm -rf data/raw/*.zip

# 压缩检查点
tar -czf checkpoints.tar.gz models/checkpoints/
```

---

## 🎯 训练监控

### WandB 配置

```bash
# 登录 WandB
wandb login

# 输入 API Key（从 https://wandb.ai/settings 获取）
```

### 实时监控

```bash
# 方法1: WandB 仪表板
https://wandb.ai/your-username/vesuvius-challenge

# 方法2: TensorBoard
tensorboard --logdir logs/ --port 6006

# 方法3: 查看日志
tail -f logs/training.log
```

### GPU 监控

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 或使用 gpustat
pip install gpustat
gpustat -i 1
```

---

## ⚡ 快速命令参考

```bash
# 激活环境
conda activate vesuvius

# 开始训练
python train.py --config configs/autodl_486.yaml

# 恢复训练
python train.py --resume models/checkpoints/checkpoint_epoch_20.pth

# 推理
python inference.py --checkpoint models/checkpoints/best_model.pth

# 查看 GPU
nvidia-smi

# 查看进程
ps aux | grep python

# 杀死进程
kill -9 [PID]
```

---

## 🎊 总结

### 您的配置优势

✅ **RTX 5090 32GB** - 比计划多 8GB  
✅ **90GB 内存** - 比计划多 26GB  
✅ **25核 CPU** - 比计划多 9核  
✅ **费用 3.03元/时** - 仅比预算高 0.53元/时  

### 优化建议

1. **使用更大的 batch size** (3 vs 2)
2. **使用更大的 patch size** (80³ vs 64³)
3. **使用标准 U-Net** (不需要轻量版)
4. **增加 num_workers** (8 vs 4)

### 预期成果

- **训练时间**: ~42小时
- **总费用**: ~127元
- **性能**: 比基线配置更好
- **目标**: Top 10%

---

**配置完美！立即开始吧！** 🚀

**有任何问题随时找我！** 💪
