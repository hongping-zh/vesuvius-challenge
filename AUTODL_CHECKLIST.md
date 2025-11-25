# 📋 AutoDL 训练检查清单

**日期**: 2025-11-23  
**任务**: DynUNet 快速验证（8 epochs）

---

## 🎯 目标

- ✅ 在 AutoDL 上成功训练 DynUNet
- ✅ 8 epochs 后 SurfaceDice > 0.65
- ✅ 验证性能提升（比 UNet3DLite +0.30）

---

## 📦 Step 1: 创建 AutoDL 实例

### 1.1 选择配置

```
机型: RTX 5090 (32GB)
镜像: PyTorch 2.0+ (推荐官方镜像)
系统盘: 50GB
数据盘: 200GB（存放数据）
```

### 1.2 启动实例

- [ ] 创建实例
- [ ] 等待启动完成
- [ ] 获取 SSH 连接信息

---

## 📤 Step 2: 上传代码

### 方法 1: Git（推荐）

```bash
# 在 AutoDL 上
cd ~
git clone <your-repo-url>
cd vesuvius-challenge
```

### 方法 2: 直接上传

```bash
# 在本地
# 使用 AutoDL 的文件上传功能
# 或者使用 scp
```

### 方法 3: 压缩上传

```powershell
# 在本地
# 1. 压缩项目（排除大文件）
tar -czf vesuvius-challenge.tar.gz \
    --exclude=data \
    --exclude=models/checkpoints* \
    --exclude=*.pth \
    vesuvius-challenge/

# 2. 上传到 AutoDL
# 3. 在 AutoDL 上解压
tar -xzf vesuvius-challenge.tar.gz
```

**检查清单**:
- [ ] 代码上传完成
- [ ] 目录结构正确
- [ ] 所有文件存在

---

## 🔧 Step 3: 安装依赖

```bash
# 进入项目目录
cd ~/vesuvius-challenge

# 安装 MONAI
pip install monai[all]==1.3.2

# 安装其他依赖
pip install connected-components-3d
pip install albumentations
pip install tifffile
pip install zarr
pip install scikit-image

# 可选：WandB（训练监控）
pip install wandb
```

**验证安装**:
```bash
python -c "import monai; print(f'MONAI {monai.__version__}')"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**检查清单**:
- [ ] MONAI 安装成功
- [ ] PyTorch 可用
- [ ] CUDA 可用
- [ ] GPU 识别正常

---

## 📥 Step 4: 下载数据

### 4.1 配置 Kaggle API

```bash
# 创建 .kaggle 目录
mkdir -p ~/.kaggle

# 上传 kaggle.json
# 方法1: 使用 AutoDL 文件上传
# 方法2: 使用 vim 创建
vim ~/.kaggle/kaggle.json
# 粘贴内容后保存

# 设置权限
chmod 600 ~/.kaggle/kaggle.json
```

### 4.2 下载数据

```bash
# 运行下载脚本
python download_data.py
```

**或者手动下载**:
```bash
# 下载比赛数据
kaggle competitions download -c vesuvius-challenge-surface-detection

# 解压
unzip vesuvius-challenge-surface-detection.zip -d data/raw/
```

**预期时间**: 1-2 小时

**检查清单**:
- [ ] Kaggle API 配置成功
- [ ] 数据下载完成
- [ ] 数据解压成功
- [ ] 数据结构正确

**验证数据**:
```bash
ls -lh data/raw/
# 应该看到 train/1, train/2, train/3 等文件夹
```

---

## 🧪 Step 5: 快速测试

### 5.1 测试 DynUNet

```bash
python test_dynunet.py
```

**预期**: 🎉 所有测试通过！

### 5.2 测试数据加载

```bash
# 创建测试脚本
python -c "
from utils.dataset import VesuviusDataset
dataset = VesuviusDataset('data/processed/train', patch_size=[96,96,96])
print(f'数据集大小: {len(dataset)}')
sample = dataset[0]
print(f'样本形状: {sample[0].shape}, {sample[1].shape}')
"
```

**检查清单**:
- [ ] DynUNet 测试通过
- [ ] 数据加载正常
- [ ] GPU 可用

---

## 🚀 Step 6: 开始训练

### 6.1 使用 tmux（推荐）

```bash
# 创建 tmux 会话
tmux new -s vesuvius

# 在 tmux 中运行训练
python train.py --config configs/autodl_dynunet_small.yaml

# 分离会话: Ctrl+B, 然后按 D
# 重新连接: tmux attach -t vesuvius
```

### 6.2 直接运行

```bash
# 后台运行
nohup python train.py --config configs/autodl_dynunet_small.yaml > train.log 2>&1 &

# 查看日志
tail -f train.log
```

### 6.3 监控训练

```bash
# 查看 GPU 使用
watch -n 1 nvidia-smi

# 查看日志
tail -f train.log

# 或者使用 WandB（如果启用）
```

**检查清单**:
- [ ] 训练开始
- [ ] GPU 使用率 > 80%
- [ ] 损失正常下降
- [ ] 没有错误

---

## 📊 Step 7: 监控进度

### 预期训练时间

| Epoch | 时间 | 累计 |
|-------|------|------|
| 1 | 30-40分钟 | 0.5-0.7h |
| 2 | 30-40分钟 | 1.0-1.3h |
| 4 | 30-40分钟 | 2.0-2.7h |
| 8 | 30-40分钟 | 4.0-5.3h |

**总时间**: 4-5 小时

### 关键指标

**Epoch 2**:
- Train Dice: > 0.40
- Val Dice: > 0.40
- SurfaceDice: > 0.30

**Epoch 4**:
- Train Dice: > 0.55
- Val Dice: > 0.50
- SurfaceDice: > 0.50

**Epoch 8**:
- Train Dice: > 0.65
- Val Dice: > 0.60
- **SurfaceDice: > 0.65** ⭐ 目标

**检查清单**:
- [ ] Epoch 2 指标正常
- [ ] Epoch 4 指标正常
- [ ] Epoch 8 达到目标

---

## 🎯 Step 8: 验证结果

### 8.1 查看最佳模型

```bash
ls -lh models/checkpoints_dynunet_small/

# 应该看到
# best_model.pth
# checkpoint_epoch_*.pth
```

### 8.2 查看训练日志

```bash
# 查看最后几行
tail -n 50 train.log

# 查看最佳分数
grep "最佳" train.log
```

### 8.3 验证性能提升

| 模型 | SurfaceDice | 提升 |
|------|-------------|------|
| UNet3DLite | 0.30-0.40 | - |
| **DynUNet** | **0.65-0.70** | **+0.30** ✅ |

**检查清单**:
- [ ] 最佳模型已保存
- [ ] SurfaceDice > 0.65
- [ ] 性能提升 > +0.30

---

## ✅ Step 9: 决策点

### 如果 SurfaceDice > 0.65 ✅

**恭喜！DynUNet 验证成功！**

**下一步**:
1. 修改配置：`epochs: 50`
2. 开始完整训练
3. 预期时间：40-50 小时
4. 预期成本：120-150 元
5. 目标：SurfaceDice > 0.75

```bash
# 修改配置
vim configs/autodl_dynunet_small.yaml
# 将 epochs: 8 改为 epochs: 50

# 或者使用 large 配置
python train.py --config configs/autodl_dynunet_large.yaml
```

### 如果 SurfaceDice < 0.65 ⚠️

**需要调试**

**可能原因**:
1. 数据问题
2. 超参数需要调整
3. 损失权重不合适

**调试步骤**:
1. 检查数据加载
2. 降低学习率
3. 调整损失权重
4. 增加训练轮数

---

## 💰 成本估算

| 项目 | 时间 | 成本 |
|------|------|------|
| 数据下载 | 1-2h | 3-6元 |
| 训练（8 epochs） | 4-5h | 12-15元 |
| **总计** | **6-7h** | **15-21元** |

**RTX 5090 价格**: ~3元/小时

---

## 📝 注意事项

### 显存管理

**如果显存不足**:
```yaml
# 修改配置
training:
  batch_size: 1
  accumulation_steps: 8

data:
  patch_size: [96, 96, 96]
```

### 数据缓存

**如果内存不足**:
```yaml
data:
  cache_rate: 0.5  # 只缓存 50%
```

### 训练稳定性

**如果训练不稳定**:
```yaml
training:
  learning_rate: 0.0001  # 降低学习率

loss:
  surface_weight: 0.3    # 降低拓扑损失权重
  topology_weight: 0.2
```

---

## 🎊 成功标准

### 必须达到（最低要求）

- ✅ 训练正常运行
- ✅ 没有错误
- ✅ 损失下降

### 目标（期望达到）

- ✅ SurfaceDice > 0.65
- ✅ Final Score > 0.55
- ✅ 性能提升 > +0.30

### 理想（最好达到）

- ✅ SurfaceDice > 0.70
- ✅ Final Score > 0.60
- ✅ 性能提升 > +0.35

---

## 📞 快速参考

### 常用命令

```bash
# 查看 GPU
nvidia-smi

# 查看进程
ps aux | grep python

# 查看日志
tail -f train.log

# tmux 操作
tmux new -s vesuvius      # 创建会话
tmux attach -t vesuvius   # 连接会话
Ctrl+B, D                 # 分离会话
tmux ls                   # 列出会话
```

### 紧急情况

**训练卡住**:
```bash
# 杀死进程
pkill -9 python

# 重新开始
python train.py --config configs/autodl_dynunet_small.yaml
```

**显存不足**:
```bash
# 清理显存
python -c "import torch; torch.cuda.empty_cache()"
```

---

**🚀 准备好了！开始 AutoDL 训练！** 💪

**预期今晚结果**: SurfaceDice > 0.65 ✅
