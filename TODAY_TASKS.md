# 📅 今天任务清单 (2025-11-23)

**目标**: 完成 DynUNet 实施并开始 AutoDL 验证

---

## ✅ Task 1: 推理测试（5分钟）

```powershell
cd C:\Users\14593\CascadeProjects\vesuvius-challenge
python inference_notebook.py
```

**预期结果**:
- ✅ 模型加载成功
- ✅ 滑动窗口推理正常
- ✅ 生成 submission.csv
- ✅ 总时间 < 3 分钟

---

## ✅ Task 2: 安装 MONAI（2分钟）

```powershell
pip install monai[all]==1.3.2
```

**验证安装**:
```powershell
python -c "import monai; print(f'MONAI {monai.__version__} 安装成功')"
```

**可选依赖**:
```powershell
pip install connected-components-3d  # 用于高级后处理
```

---

## ✅ Task 3: 测试 DynUNet（5分钟）

```powershell
python test_dynunet.py
```

**预期输出**:
```
测试 1: 导入 DynUNet
✅ DynUNet 导入成功

测试 2: 创建 DynUNet 模型
✅ DynUNet 创建成功

测试 3: DynUNet 前向传播
✅ 前向传播测试通过

测试 4: 加载 DynUNet 配置
✅ 配置文件加载成功

🎉 所有测试通过！
```

---

## ✅ Task 4: 本地快速验证（可选，10分钟）

**使用测试数据验证训练流程**:

```powershell
# 使用测试配置（小数据集）
python train.py --config configs/test.yaml
```

**目的**: 确保 DynUNet 能正常训练

---

## 🚀 Task 5: AutoDL 快速验证（4-5小时，12-15元）

### 5.1 准备工作

**在 AutoDL 上**:

```bash
# 1. 创建实例
# - 选择 RTX 5090 (32GB)
# - 选择 PyTorch 镜像

# 2. 上传代码
# 使用 git 或直接上传

# 3. 安装依赖
pip install monai[all]==1.3.2
pip install connected-components-3d
pip install wandb  # 可选
```

### 5.2 下载真实数据

```bash
# 配置 Kaggle API
mkdir -p ~/.kaggle
# 上传 kaggle.json

# 下载数据
python download_data.py
```

**预期时间**: 1-2 小时

### 5.3 开始训练

```bash
# 使用 tmux 保持会话
tmux new -s vesuvius

# 开始训练（8 epochs 快速验证）
python train.py --config configs/autodl_dynunet_small.yaml
```

**预期结果**:
- 训练时间: 4-5 小时
- Fragment 1 SurfaceDice: **>0.65** ✅
- 成本: ~12-15 元

### 5.4 监控训练

```bash
# 查看日志
tail -f logs/train.log

# 或者使用 WandB（如果启用）
```

---

## 📊 预期性能

### 8 Epochs 后

| 指标 | 目标值 | 说明 |
|------|--------|------|
| Train Loss | < 1.0 | 下降趋势 |
| Val Loss | < 1.5 | 下降趋势 |
| Train Dice | > 0.60 | 基础指标 |
| Val Dice | > 0.60 | 基础指标 |
| **SurfaceDice@τ** | **> 0.65** | 关键指标 ✅ |
| VOI_score | > 0.50 | 拓扑指标 |
| TopoScore | > 0.40 | 拓扑指标 |
| **Final Score** | **> 0.55** | 综合指标 |

### 如果达到目标

**继续完整训练**:
```bash
# 修改配置：epochs: 50
vim configs/autodl_dynunet_small.yaml

# 或者使用 large 配置
python train.py --config configs/autodl_dynunet_large.yaml
```

---

## ⚠️ 常见问题

### Q1: MONAI 安装失败

```powershell
# 尝试不带 [all]
pip install monai==1.3.2

# 或者使用清华镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple monai[all]==1.3.2
```

### Q2: 显存不足

**修改配置**:
```yaml
training:
  batch_size: 1          # 降低 batch size
  accumulation_steps: 8  # 增加梯度累积

data:
  patch_size: [96, 96, 96]  # 降低 patch size
```

### Q3: 训练速度慢

**检查**:
- ✅ 使用 GPU（不是 CPU）
- ✅ cache_rate: 1.0（缓存数据）
- ✅ num_workers: 4（多线程加载）
- ✅ pin_memory: true

### Q4: 数据下载失败

**手动下载**:
1. 访问 Kaggle 比赛页面
2. 下载数据集
3. 上传到 AutoDL
4. 解压到 `data/raw/`

---

## 📋 检查清单

### 本地准备

- [ ] 推理测试完成
- [ ] MONAI 安装成功
- [ ] DynUNet 测试通过
- [ ] 配置文件检查

### AutoDL 准备

- [ ] 创建 AutoDL 实例
- [ ] 上传代码
- [ ] 安装依赖
- [ ] 配置 Kaggle API
- [ ] 下载真实数据

### 训练验证

- [ ] 开始训练（8 epochs）
- [ ] 监控训练进度
- [ ] 检查 SurfaceDice > 0.65
- [ ] 保存最佳模型

---

## 🎯 成功标准

### 最低目标（必须达到）

- ✅ DynUNet 能正常训练
- ✅ 没有报错
- ✅ 损失正常下降

### 目标（期望达到）

- ✅ SurfaceDice > 0.65
- ✅ Final Score > 0.55
- ✅ 比 UNet3DLite 提升 +0.30

### 理想目标（最好达到）

- ✅ SurfaceDice > 0.70
- ✅ Final Score > 0.60
- ✅ 比 UNet3DLite 提升 +0.35

---

## 💰 成本估算

| 阶段 | 时间 | 成本 |
|------|------|------|
| 本地测试 | 30分钟 | 0元 |
| 数据下载 | 1-2小时 | 0元 |
| 训练（8 epochs） | 4-5小时 | 12-15元 |
| **总计** | **~6小时** | **~12-15元** |

---

## 📞 快速参考

### 关键文件

```
models/
├── dynunet.py              ✅ 已创建

configs/
├── autodl_dynunet_small.yaml  ✅ 已创建

utils/
├── topology_refine.py      ✅ 已创建

test_dynunet.py             ✅ 已创建
```

### 关键命令

```powershell
# 本地测试
python test_dynunet.py
python inference_notebook.py

# AutoDL 训练
python train.py --config configs/autodl_dynunet_small.yaml
```

### 关键文档

- [`DYNUNET_GUIDE.md`](DYNUNET_GUIDE.md) - 完整指南
- [`CRITICAL_IMPROVEMENTS.md`](CRITICAL_IMPROVEMENTS.md) - 改进清单
- [`TEST_REPORT.md`](TEST_REPORT.md) - 测试结果

---

## 🎊 今天目标

**核心目标**: 
1. ✅ 完成本地测试
2. ✅ 在 AutoDL 上开始训练
3. ✅ 验证 DynUNet 性能提升

**如果一切顺利**:
- 今晚能看到 8 epochs 的结果
- SurfaceDice > 0.65
- 确认 DynUNet 有效

**明天计划**:
- 如果验证成功，开始 50 epochs 完整训练
- 如果需要调整，修改配置后重新训练

---

**🚀 开始吧！冲击 Top 10%！** 💪
