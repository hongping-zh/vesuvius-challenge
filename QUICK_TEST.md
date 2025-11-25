# 🚀 快速测试指南

**目标**: 5-10 分钟内验证完整训练流程

---

## Step 1: 创建测试数据

```powershell
python create_sample_data.py
```

**预期输出**:
```
============================================================
创建测试数据
============================================================

📦 创建训练数据...
   ✓ 训练体积: (128, 128, 128)
   ✓ 训练掩码: (128, 128, 128)
   ✓ 掩码覆盖率: 15.23%

📦 创建验证数据...
   ✓ 验证体积: (128, 128, 128)
   ✓ 验证掩码: (128, 128, 128)
   ✓ 掩码覆盖率: 12.45%

📦 创建测试数据...
   ✓ 测试体积: (128, 128, 128)

============================================================
✅ 测试数据创建完成！
============================================================
```

**生成的文件**:
```
data/processed/
├── train/
│   ├── volume.npy  (~8 MB)
│   └── mask.npy    (~8 MB)
├── val/
│   ├── volume.npy  (~8 MB)
│   └── mask.npy    (~8 MB)
└── test/
    └── volume.npy  (~8 MB)
```

---

## Step 2: 测试基础训练

```powershell
python train.py --config configs/test.yaml
```

**配置说明**:
- 模型: UNet3DLite (轻量级)
- Epochs: 3
- Batch size: 1
- Patch size: 32×32×32
- 损失: DiceBCE (标准)
- 不启用 WandB

**预期输出**:
```
使用设备: cpu
使用标准 DiceBCE 损失函数

Epoch 1/3
------------------------------------------------------------
Training: 100%|████████| 10/10 [00:30<00:00]
Validation: 100%|████████| 5/5 [00:10<00:00]

Train Loss: 0.6234, Train Dice: 0.4521
Val Loss: 0.5823, Val Dice: 0.4892
Learning Rate: 0.001000

✅ 保存最佳模型 (Dice: 0.4892)

Epoch 2/3
...

============================================================
训练完成！最佳 Dice: 0.5234
============================================================
```

**预期时间**: 3-5 分钟（CPU）

---

## Step 3: 测试拓扑优化训练

```powershell
python train.py --config configs/test_optimized.yaml
```

**配置说明**:
- 模型: UNet3DLite
- Epochs: 3
- 损失: **VesuviusCompositeLoss** ⭐
- 评估: **Vesuvius Metrics** ⭐
- 后处理: **启用** ⭐

**预期输出**:
```
使用设备: cpu
使用 Vesuvius 组合损失函数
使用 Vesuvius 评估指标
启用拓扑感知后处理

Epoch 1/3
------------------------------------------------------------
Training: 100%|████████| 10/10 [00:45<00:00]

Train Loss: 0.5271, Train Dice: 0.4823
  Loss Components: Dice=0.4997, BCE=0.8061, Surface=0.2545, 
                   Centerline=0.9989, Topology=0.0257

Validation: 100%|████████| 5/5 [00:20<00:00]

Val Loss: 0.4823, Val Dice: 0.5234
  Vesuvius Metrics: SurfaceDice=0.5612, VOI=0.5123, 
                    Topo=0.4891, Final=0.5201

Learning Rate: 0.001000

✅ 保存最佳模型 (Score: 0.5201)
```

**预期时间**: 5-8 分钟（CPU，因为有后处理）

**关键验证**:
- ✅ 损失组件正常显示
- ✅ Vesuvius 指标正常计算
- ✅ 后处理正常运行
- ✅ Final Score 用于保存最佳模型

---

## Step 4: 测试推理

```powershell
python inference_notebook.py
```

**预期输出**:
```
============================================================
Vesuvius Challenge Inference
============================================================

📥 加载模型...
✓ 模型加载完成
  Epoch: 3
  Best Score: 0.5201

📥 加载测试数据...
✓ 测试数据加载完成
  形状: (128, 128, 128)
  范围: [0.0234, 0.9876]

🔮 开始推理...
总共 27 个 patches
推理中: 100%|████████| 7/7 [01:30<00:00]
✓ 推理完成
  预测范围: [0.0123, 0.9234]

🔧 后处理...
✓ 后处理完成
  预测像素: 245678 / 2097152
  覆盖率: 11.72%

📤 生成提交文件...
✓ 提交文件已生成: submission.csv

============================================================
✅ 推理完成！
============================================================
总耗时: 0h 2m
时间限制: 9h (剩余: 8h 58m)
✓ 在时间限制内完成
```

**预期时间**: 2-3 分钟

---

## ✅ 测试通过标准

### 基础训练测试

- ✅ 训练循环正常运行
- ✅ 损失逐渐下降
- ✅ Dice 逐渐提升
- ✅ 检查点正常保存
- ✅ 没有错误或警告

### 拓扑优化测试

- ✅ 所有损失组件正常
- ✅ Vesuvius 指标正常计算
- ✅ 后处理正常运行
- ✅ Final Score 正常显示
- ✅ 使用 Final Score 保存模型

### 推理测试

- ✅ 模型加载成功
- ✅ 滑动窗口推理正常
- ✅ 后处理正常
- ✅ 生成提交文件
- ✅ 时间在限制内

---

## 🐛 常见问题

### 问题 1: 显存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决**: 使用 CPU 测试
```yaml
# 在配置文件中不需要修改
# train.py 会自动检测设备
```

### 问题 2: 训练很慢

**症状**: 每个 epoch 超过 5 分钟

**原因**: 正常，CPU 训练较慢

**优化**:
```yaml
# configs/test.yaml
data:
  patch_size: [24, 24, 24]  # 更小的 patch
training:
  num_workers: 0  # 禁用多进程
```

### 问题 3: 后处理报错

**症状**: `ImportError: cannot import name 'remove_small_objects'`

**解决**:
```powershell
pip install scikit-image
```

---

## 📊 预期性能

### 测试数据（合成数据）

| 指标 | 基础训练 | 拓扑优化 |
|------|---------|---------|
| Train Dice | 0.50-0.55 | 0.48-0.52 |
| Val Dice | 0.48-0.53 | 0.50-0.55 |
| Final Score | - | 0.50-0.55 |

**注意**: 合成数据性能不代表真实数据！

---

## 🎯 测试完成后

### 如果所有测试通过 ✅

**下一步**:
1. 下载真实比赛数据
2. 租用 AutoDL 5090 RTX
3. 使用 `configs/autodl_486_optimized.yaml` 训练
4. 预期 42 小时，~127 元

### 如果有测试失败 ❌

**排查步骤**:
1. 查看错误信息
2. 检查依赖包是否安装
3. 验证数据是否正确生成
4. 查看日志文件

---

## 💡 提示

### 加速测试

```yaml
# 最小测试配置
training:
  epochs: 1
  batch_size: 1
data:
  patch_size: [24, 24, 24]
augmentation:
  random_flip: false
  random_rotation: 0
  elastic_deformation: false
```

### 完整验证

```yaml
# 完整功能测试
training:
  epochs: 5
loss:
  type: 'vesuvius_composite'
evaluation:
  use_vesuvius_metrics: true
postprocessing:
  enabled: true
```

---

## 📝 测试清单

- [ ] Step 1: 创建测试数据
- [ ] Step 2: 基础训练测试
- [ ] Step 3: 拓扑优化测试
- [ ] Step 4: 推理测试
- [ ] 验证所有输出正常
- [ ] 检查生成的文件
- [ ] 准备真实数据训练

---

**完成所有测试后，您就可以放心地租用 AutoDL 进行完整训练了！** 🚀
