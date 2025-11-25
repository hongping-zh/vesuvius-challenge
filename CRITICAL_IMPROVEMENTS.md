# 🚨 关键改进清单 - 上真实数据前必须完成

**优先级**: ⭐⭐⭐⭐⭐ **最高**  
**状态**: ⚠️ **必须在真实训练前完成**  
**预计影响**: +0.15~0.25 Final Score

---

## ⚠️ 当前问题诊断

### 问题 1: 模型容量严重不足 🔴

**现状**:
- 使用 `UNet3DLite` (1.46M 参数)
- Base channels: 16
- 在合成数据上够用

**真实挑战**:
- Scroll volume 巨大：65+ 层、5000×5000+ xy
- 噪声极强
- 墨迹极稀疏：**<0.1% 正像素**
- Lite 版学不到足够深层特征

**后果**:
- ❌ Val Dice 可能还行，但 Vesuvius Metrics 为 0（已验证）
- ❌ 无法捕捉复杂的 3D 拓扑结构
- ❌ 泛化能力差

### 问题 2: Loss 权重不适合真实数据 🟡

**现状**:
```yaml
dice_weight: 0.3
bce_weight: 0.2
surface_weight: 0.25
centerline_weight: 0.15
topology_weight: 0.1
```

**问题**:
- Surface/Topology 能降，但 Dice 拉太狠
- 导致 TopoScore = 0
- CenterlineLoss 在墨迹检测中用处不大（不是血管）

### 问题 3: 合成数据太简单 🟡

**现状**:
- 简单的圆柱形结构
- Val Dice 高但 Vesuvius Metrics = 0

**必须**:
- 使用真实 fragment 数据
- 建立可靠的交叉验证

---

## 🎯 必须改进项（优先级排序）

### 🔴 P0: 升级模型架构（最关键）

#### 推荐方案（按优先级）

**方案 1: DynUNet（MONAI）⭐ 推荐**
```python
from monai.networks.nets import DynUNet

model = DynUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    kernel_size=[3, 3, 3, 3, 3],
    strides=[1, 2, 2, 2, 2],
    upsample_kernel_size=[2, 2, 2, 2],
    filters=[64, 128, 256, 512, 1024],  # 更大容量
    dropout=0.1,
    deep_supervision=True,  # 关键！
    deep_supr_num=3
)
```

**优势**:
- ✅ 自适应配置
- ✅ 上届 SOTA 标配
- ✅ Deep supervision 提升收敛
- ✅ 参数量：~50M（合理）

**方案 2: nnUNet-3D**
```python
# 使用 nnUNet 框架
# 自动配置最优架构
```

**优势**:
- ✅ 自动调参
- ✅ 医学图像分割 SOTA
- ✅ 鲁棒性强

**方案 3: SwinUNETR（Transformer-based）⭐⭐**
```python
from monai.networks.nets import SwinUNETR

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=1,
    feature_size=48,
    use_checkpoint=True,  # 节省显存
    spatial_dims=3
)
```

**优势**:
- ✅ 上届很多 Top10 在用
- ✅ 全局感受野
- ✅ 更好的长距离依赖

**方案 4: UNETR（ViT-based）**
```python
from monai.networks.nets import UNETR

model = UNETR(
    in_channels=1,
    out_channels=1,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    spatial_dims=3
)
```

#### 配置建议

**最低要求**:
- Base channels: 64（当前 16）
- Deep supervision: 启用
- 参数量: 30M~100M

**推荐配置（RTX 5090）**:
- Base channels: 128
- Patch size: 96×96×96 或 128×128×128
- Batch size: 2~4

#### 实现步骤

1. **安装 MONAI**
```bash
pip install monai
```

2. **创建新模型文件**
```
models/
├── unet3d.py          # 保留（备用）
├── dynunet.py         # 新增 ⭐
├── swinunetr.py       # 新增
└── unetr.py           # 新增
```

3. **修改 train.py**
```python
# 支持多种模型
if config['model']['type'] == 'dynunet':
    from models.dynunet import DynUNet
    model = DynUNet(...)
elif config['model']['type'] == 'swinunetr':
    from models.swinunetr import SwinUNETR
    model = SwinUNETR(...)
```

---

### 🟠 P1: 重新设计 Loss 策略

#### 推荐方案（来自上届 winner）

**阶段 1: 学习"哪里有墨"（Epoch 1~20）**
```yaml
loss:
  type: 'dice_focal'  # 或 dice_bce
  dice_weight: 0.5
  focal_weight: 0.5
  focal_alpha: 0.25
  focal_gamma: 2.0
```

**阶段 2: 加入拓扑约束（Epoch 21~50）**
```yaml
loss:
  type: 'vesuvius_composite'
  dice_weight: 0.4
  bce_weight: 0.2
  surface_weight: 0.2      # 降低
  centerline_weight: 0.0   # 关闭！
  topology_weight: 0.2     # 提高
  cldice_weight: 0.1       # 新增！
```

#### 关键改进

1. **关闭 CenterlineLoss**
   - 墨迹不是血管
   - 不适用于本任务

2. **添加 ClDice（Centerline Dice）**
   - 专门针对细长结构
   - 上届很多队伍用

3. **动态权重调度**
```python
def get_loss_weights(epoch, total_epochs):
    if epoch < total_epochs * 0.4:
        # 前 40%：只学基础
        return {
            'dice': 0.5,
            'focal': 0.5,
            'surface': 0.0,
            'topology': 0.0
        }
    else:
        # 后 60%：加入拓扑
        return {
            'dice': 0.4,
            'focal': 0.2,
            'surface': 0.2,
            'topology': 0.2
        }
```

---

### 🟠 P2: 真实数据准备（最影响分数）

#### 立即执行

```powershell
# 1. 下载真实数据
python download_data.py

# 2. 建立交叉验证
python create_cv_splits.py --n_folds 5
```

#### 数据策略

**交叉验证方案**:
- **5-Fold CV**（按 fragment 分）
- 或 **LOFO**（Leave-One-Fragment-Out）

**示例分割**:
```
Fold 1: train=[1,2,3], val=[4]
Fold 2: train=[1,2,4], val=[3]
Fold 3: train=[1,3,4], val=[2]
Fold 4: train=[2,3,4], val=[1]
Fold 5: train=[1,2,3,4], val=[5]
```

#### 数据采样策略

**问题**: 墨迹极不平衡（<0.1% 正像素）

**解决**: Ink-only positive sampling
```python
# 只采样包含 ink 的 patch
# 负样本比例控制在 20~30%

positive_ratio = 0.7  # 70% 包含墨迹
negative_ratio = 0.3  # 30% 纯背景
```

---

### 🟡 P3: 推理后处理迭代

#### 当前状态
✅ 已实现基础拓扑后处理

#### 必须添加

**1. 多尺度预测**
```python
# 预测 11~21 层 surface volume
# 然后 argmax 或 soft vote
predictions = []
for depth in range(11, 22):
    pred = model(volume[:, :, depth-5:depth+6])
    predictions.append(pred)

final = soft_vote(predictions)
```

**2. 高级形态学处理**
```python
# Connected Component Analysis
# 面积过滤
# 孔洞填充
# Persistence-based topology simplification
```

**3. 自适应阈值**
```python
# 不同 fragment 不同阈值
# 在 0.2~0.5 之间扫描
thresholds = {
    'fragment_1': 0.3,
    'fragment_2': 0.4,
    'fragment_3': 0.35,
}
```

---

## 🚀 高回报、中等难度改进

### 改进 1: 多通道输入 ⭐⭐⭐

**当前**: 只有 raw intensity (1 通道)

**改进**: 添加特征通道（3~9 通道）
```python
# 1. Gradient (3 通道)
grad_x = np.gradient(volume, axis=0)
grad_y = np.gradient(volume, axis=1)
grad_z = np.gradient(volume, axis=2)

# 2. LoG (Laplacian of Gaussian)
log = ndimage.gaussian_laplace(volume, sigma=1.0)

# 3. Hessian eigenvalues
hessian = compute_hessian(volume)

# 4. Local entropy
entropy = local_entropy(volume, kernel_size=5)

# 组合
input_channels = np.stack([
    volume,      # raw
    grad_x,      # gradient
    grad_y,
    grad_z,
    log,         # LoG
    hessian[0],  # eigenvalue 1
    hessian[1],  # eigenvalue 2
    hessian[2],  # eigenvalue 3
    entropy      # local entropy
], axis=0)  # (9, D, H, W)
```

**预期提升**: +0.05~0.1 Final Score  
**实现难度**: 低

### 改进 2: 增强数据增强 ⭐⭐

**添加**:
```python
augmentation:
  # 3D 专用
  random_rotate_90: true
  elastic_deform: true
  elastic_alpha: [100, 200]
  elastic_sigma: [10, 20]
  
  # 强度增强
  gaussian_noise: true
  noise_std: 0.1
  contrast_adjust: true
  contrast_range: [0.8, 1.2]
  
  # 空间增强
  random_scale: true
  scale_range: [0.9, 1.1]
```

**预期提升**: 防止过拟合  
**实现难度**: 低

### 改进 3: 更大 Patch Size ⭐⭐⭐

**当前**: 32×32×32 或 64×64×64

**改进**: 96×96×96 或 128×128×128
```yaml
data:
  patch_size: [96, 96, 96]  # RTX 5090 能吃
```

**优势**: 更多上下文信息  
**预期提升**: +0.03~0.05  
**实现难度**: 中（需要调整显存）

### 改进 4: 优化器升级 ⭐

**当前**: AdamW + Cosine

**改进**: AdamW + CosineAnnealingWarmupRestarts
```python
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=200,
    cycle_mult=1.0,
    max_lr=0.001,
    min_lr=0.00001,
    warmup_steps=50,
    gamma=0.5
)
```

**预期提升**: 更稳定收敛  
**实现难度**: 低

### 改进 5: Ensemble ⭐⭐⭐

**策略**:
```python
# 1. 不同 backbone
models = [
    DynUNet(...),
    SwinUNETR(...),
    UNETR(...)
]

# 2. 不同 fold
folds = [0, 1, 2, 3, 4]

# 3. Flip TTA
predictions = []
for model in models:
    for fold in folds:
        pred = model(volume)
        pred_flip = model(flip(volume))
        predictions.append(pred)
        predictions.append(flip_back(pred_flip))

final = np.mean(predictions, axis=0)
```

**预期提升**: +0.03~0.08  
**实现难度**: 中

### 改进 6: 预训练权重 ⭐⭐⭐⭐

**来源**: 上届 Ink Detection 预训练的 3D UNet

**Kaggle Dataset**:
- 搜索 "vesuvius pretrained"
- 下载权重

**使用**:
```python
# 加载预训练权重
checkpoint = torch.load('pretrained_weights.pth')
model.load_state_dict(checkpoint, strict=False)
```

**预期提升**: 加速 2~3 倍收敛  
**实现难度**: 低

---

## 📅 时间与成本最优路径（穷人友好版）

### Day 1（明天）⏰ 2~4h / 💰 0元

**任务**:
- ✅ 跑推理测试
- ✅ 用 train/1 小 fragment 本地 CV
- ✅ 确认滑动窗口推理 <9h

**输出**:
- 推理流程验证
- 时间估算

### Day 2~3 ⏰ 8~15h / 💰 20~30元

**任务**:
- ✅ 下载全部数据
- ✅ 建 3~5 fold
- ✅ 跑 5~8 epochs 小模型验证
- ✅ UNet3DLite → DynUNet

**目标**:
- 真实 Vesuvius Score >0.4
- 验证新模型架构

### Day 4~7 ⏰ 40~60h / 💰 120~180元

**任务**:
- ✅ 中大模型（DynUNet/SwinUNETR）
- ✅ 损失调权
- ✅ 后处理迭代
- ✅ 训练 30~50 epochs

**目标**:
- 本地 Final Score 0.68~0.75+

### Day 8+ ⏰ +20h / 💰 50元

**任务**:
- ✅ Ensemble 3~5 个模型
- ✅ 提交

**目标**:
- 冲 LB Top50 → Top10

---

## 📋 明天必做清单

### ⭐⭐⭐⭐⭐ 最高优先级

- [ ] 完成推理测试
- [ ] 安装 MONAI：`pip install monai`
- [ ] 创建 `models/dynunet.py`
- [ ] 修改 `train.py` 支持 DynUNet
- [ ] 下载真实数据：`python download_data.py`
- [ ] 创建交叉验证脚本：`create_cv_splits.py`

### ⭐⭐⭐⭐ 高优先级

- [ ] 修改 Loss 配置（关闭 CenterlineLoss）
- [ ] 添加动态权重调度
- [ ] 实现 ink-only sampling
- [ ] 添加多通道输入

### ⭐⭐⭐ 中优先级

- [ ] 增强数据增强
- [ ] 增大 patch size
- [ ] 优化器升级
- [ ] 搜索预训练权重

---

## 💰 成本重新估算

| 阶段 | 时间 | 成本 | 备注 |
|------|------|------|------|
| 推理测试 | 3分钟 | 0元 | 明天 |
| 数据下载 | 1-3h | 0元 | Day 1 |
| 快速验证（DynUNet） | 8-15h | 20-30元 | Day 2-3 |
| 完整训练（中大模型） | 40-60h | 120-180元 | Day 4-7 |
| Ensemble | 20h | 50元 | Day 8+ |
| **总计** | **~80h** | **~200-260元** | |

**预期排名**: Top 10% (0.70-0.75 Final Score)

---

## 🎯 成功标准

### 最低目标
- Local CV Final Score: 0.65+
- Public LB: 0.60+
- 排名: Top 50%

### 目标
- Local CV Final Score: 0.70+
- Public LB: 0.68+
- 排名: Top 20%

### 理想目标
- Local CV Final Score: 0.75+
- Public LB: 0.72+
- 排名: **Top 10%** 🏆

---

## 📞 快速参考

### 关键文件

```
models/
├── dynunet.py          # 明天创建 ⭐
├── swinunetr.py        # 可选
└── unetr.py            # 可选

configs/
├── autodl_486_dynunet.yaml    # 明天创建 ⭐
└── autodl_486_optimized.yaml  # 需要更新

scripts/
├── create_cv_splits.py        # 明天创建 ⭐
└── download_data.py           # 已有
```

### 关键命令

```bash
# 安装 MONAI
pip install monai

# 下载数据
python download_data.py

# 创建交叉验证
python create_cv_splits.py --n_folds 5

# 训练（新模型）
python train.py --config configs/autodl_486_dynunet.yaml
```

---

## ⚠️ 警告

### 不要做的事

1. ❌ 不要直接用 UNet3DLite 训练真实数据
2. ❌ 不要跳过交叉验证
3. ❌ 不要忽略 ink-only sampling
4. ❌ 不要在合成数据上过度调参

### 必须做的事

1. ✅ 升级到中大型模型
2. ✅ 使用真实数据建 CV
3. ✅ 重新调整 Loss 权重
4. ✅ 实现 ink-only sampling
5. ✅ 迭代后处理

---

**🚨 这些改进是成功的关键！不做的话很难进 Top 10%！**

**明天优先级**: 推理测试 → 安装 MONAI → 创建 DynUNet → 下载数据

**预计工作量**: 4-6 小时

**预期提升**: +0.15~0.25 Final Score

---

**📅 创建时间**: 2025-11-22 19:50  
**📝 更新**: 基于测试结果和上届经验

**🎯 目标**: Top 10% 🏆
