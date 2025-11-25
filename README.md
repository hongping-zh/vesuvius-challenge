# Vesuvius Challenge - Surface Detection

**比赛**: [Vesuvius Challenge - Surface Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection)  
**任务**: 虚拟展开古代赫库兰尼姆卷轴，检测表面和墨迹  
**计算资源**: AutoDL RTX 5090 (32GB)

---

## 🚨 重要提示

**⚠️ 上真实数据前必读**: [`CRITICAL_IMPROVEMENTS.md`](CRITICAL_IMPROVEMENTS.md)

**关键问题**:
- ❌ 当前 UNet3DLite 容量不足，无法处理真实数据
- ❌ 必须升级到 DynUNet/SwinUNETR
- ❌ Loss 权重需要重新设计
- ✅ 预期提升：+0.15~0.25 Final Score

**立即行动**: 查看 [`TOMORROW_TASKS.md`](TOMORROW_TASKS.md) 获取详细计划

---

## 🎯 项目概述

本项目提供了一个完整的 Vesuvius Challenge 参赛方案，针对 **AutoDL RTX 5090** 优化，适合预算有限的参赛者。

### 核心特性

- ✅ **3D U-Net** 模型（医学图像分割标准架构）
- ✅ **混合精度训练** (FP16) - 节省显存
- ✅ **梯度累积** - 等效大 batch size
- ✅ **检查点保存** - 随时暂停恢复
- ✅ **WandB 监控** - 实时训练可视化
- ✅ **成本优化** - 最大化利用有限资源

---

## 📁 项目结构

```
vesuvius-challenge/
├── configs/                # 配置文件
│   ├── baseline.yaml      # 基线配置
│   └── lite.yaml          # 轻量级配置
├── models/                # 模型定义
│   ├── unet3d.py         # 3D U-Net
│   └── checkpoints/      # 模型检查点
├── utils/                 # 工具函数
│   ├── dataset.py        # 数据加载
│   ├── losses.py         # 损失函数
│   ├── metrics.py        # 评估指标
│   └── augmentation.py   # 数据增强
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   └── processed/        # 预处理数据
├── train.py              # 训练脚本
├── inference.py          # 推理脚本
├── download_data.py      # 数据下载
├── preprocess.py         # 数据预处理
├── requirements.txt      # 依赖列表
├── QUICK_START.md        # 快速启动指南
├── COMPETITION_PLAN.md   # 完整参赛方案
└── README.md             # 本文件
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/vesuvius-challenge.git
cd vesuvius-challenge

# 安装依赖
pip install -r requirements.txt

# 配置 Kaggle API
mkdir -p ~/.kaggle
# 将 kaggle.json 放到 ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. 下载数据

```bash
python download_data.py
```

### 3. 训练模型

```bash
# 基线模型
python train.py --config configs/baseline.yaml

# 轻量级模型（更省显存）
python train.py --config configs/lite.yaml
```

### 4. 生成提交

```bash
python inference.py \
  --checkpoint models/checkpoints/best_model.pth \
  --output submissions/submission.csv
```

---

## 💰 成本估算

### AutoDL RTX 5090 费用

**配置**:
- GPU: RTX 5090 (24GB) × 1
- CPU: 16核
- 内存: 64GB
- 存储: 500GB SSD

**费用**: ~2.5元/小时

**训练计划**:
- 快速验证: 2小时 × 2.5元 = 5元
- 基线训练: 20小时 × 2.5元 = 50元
- 优化训练: 30小时 × 2.5元 = 75元
- 推理: 5小时 × 2.5元 = 12.5元

**总计**: ~150元/月

### 省钱技巧

1. **数据预处理在 Kaggle Notebook 完成**（免费 30h/week）
2. **仅在训练时租用 GPU**
3. **使用检查点随时暂停恢复**
4. **使用轻量级模型**

---

## 📊 模型架构

### 3D U-Net

```
Input: (B, 1, 64, 64, 64)
↓
Encoder (5 levels)
  - Conv3D + BN + ReLU
  - MaxPool3D
↓
Bottleneck
↓
Decoder (5 levels)
  - Upsample
  - Concat with skip connection
  - Conv3D + BN + ReLU
↓
Output: (B, 1, 64, 64, 64)
```

**参数量**: ~15M (UNet3D) / ~3M (UNet3DLite)  
**显存需求**: ~18-20GB (UNet3D) / ~12-14GB (UNet3DLite)

---

## 🎯 训练配置

### 基线配置

```yaml
model:
  type: 'unet3d_lite'
  base_channels: 32

data:
  patch_size: [64, 64, 64]

training:
  batch_size: 2
  accumulation_steps: 4  # 等效 batch_size = 8
  epochs: 50
  learning_rate: 0.0001
  mixed_precision: true
```

### 优化技巧

1. **混合精度训练** - 节省 50% 显存
2. **梯度累积** - 等效大 batch size
3. **Patch-based 训练** - 处理大体积数据
4. **数据增强** - 提升泛化能力
5. **学习率调度** - Cosine Annealing

---

## 📈 训练监控

### WandB 集成

```bash
# 登录 WandB
wandb login

# 训练时自动记录
python train.py --config configs/baseline.yaml

# 访问仪表板
https://wandb.ai/your-username/vesuvius-challenge
```

### 关键指标

- **Dice Score**: 主要评估指标
- **IoU**: 交并比
- **Train/Val Loss**: 训练/验证损失
- **Learning Rate**: 学习率变化

---

## 🏆 优化路线图

### Phase 1: 基线（Week 1）
- [x] 环境配置
- [x] 数据下载
- [x] 基线模型训练
- [ ] 首次提交

### Phase 2: 优化（Week 2）
- [ ] 数据增强优化
- [ ] 超参数调优
- [ ] 模型架构改进

### Phase 3: 进阶（Week 3）
- [ ] 多模型集成
- [ ] 伪标签（Semi-supervised）
- [ ] 后处理优化

### Phase 4: 冲刺（Week 4）
- [ ] 最终优化
- [ ] 多次提交
- [ ] 冲击 Top 10%

---

## 📚 参考资源

### 论文
- [3D U-Net](https://arxiv.org/abs/1606.06650)
- [nnU-Net](https://arxiv.org/abs/1809.10486)
- [Vesuvius Challenge 技术报告](https://scrollprize.org/)

### 代码
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [MONAI](https://github.com/Project-MONAI/MONAI)

### 数据
- [Vesuvius Challenge 官网](https://scrollprize.org/)
- [Kaggle 数据集](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/data)

---

## 🔧 常见问题

### Q: 显存不足怎么办？

**A**: 
1. 减小 batch_size 到 1
2. 减小 patch_size 到 [48, 48, 48]
3. 使用 UNet3DLite 模型
4. 开启梯度检查点

### Q: 训练速度太慢？

**A**:
1. 增加 num_workers
2. 使用混合精度（已默认开启）
3. 减少数据增强
4. 使用更小的模型

### Q: 如何恢复训练？

**A**:
```bash
python train.py --resume models/checkpoints/checkpoint_epoch_20.pth
```

---

## 📞 支持

有问题？

1. 查看 [QUICK_START.md](QUICK_START.md)
2. 查看 [COMPETITION_PLAN.md](COMPETITION_PLAN.md)
3. 提交 Issue
4. 联系作者

---

## 📄 许可证

MIT License

---

## 🙏 致谢

- Vesuvius Challenge 组织者
- Kaggle 社区
- AutoDL 平台

---

**祝您比赛顺利！冲击 Top 10%！** 🏆
