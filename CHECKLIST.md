# Vesuvius Challenge - AutoDL 486机 启动清单

**配置**: RTX 5090 (32GB) + 25核 CPU + 90GB 内存  
**费用**: ￥3.03/时

---

## ✅ 启动前检查清单

### 1. 本地准备 (在租用前完成)

- [ ] 下载 kaggle.json
  - 访问 https://www.kaggle.com/settings
  - 点击 "Create New API Token"
  - 保存 kaggle.json

- [ ] 准备代码
  - [ ] 所有代码文件已准备
  - [ ] 配置文件已优化 (autodl_486.yaml)
  - [ ] 脚本文件已创建

- [ ] 创建 GitHub 仓库（可选但推荐）
  ```bash
  cd vesuvius-challenge
  git init
  git add .
  git commit -m "Initial commit"
  git remote add origin https://github.com/YOUR_USERNAME/vesuvius-challenge.git
  git push -u origin main
  ```

---

## 🚀 AutoDL 配置流程

### Step 1: 租用实例 (5分钟)

- [ ] 访问 https://www.autodl.com/market/list
- [ ] 选择 486机 RTX 5090
- [ ] 确认配置:
  - GPU: RTX 5090 (32GB)
  - CPU: 25核
  - 内存: 90GB
  - 数据盘: 50GB
- [ ] 点击"租用"
- [ ] 记录 SSH 信息

### Step 2: SSH 登录 (2分钟)

```bash
# AutoDL 提供的 SSH 命令
ssh -p [端口] root@[IP地址]

# 例如:
ssh -p 12345 root@region-1.autodl.com
```

- [ ] 成功登录
- [ ] 记录登录信息

### Step 3: 环境配置 (15分钟)

```bash
# 1. 创建工作目录
mkdir -p /root/projects
cd /root/projects

# 2. 克隆代码
# 方法A: 从 GitHub
git clone https://github.com/YOUR_USERNAME/vesuvius-challenge.git

# 方法B: 从本地上传
# 使用 AutoDL 文件上传功能或 scp

cd vesuvius-challenge

# 3. 创建 Conda 环境
conda create -n vesuvius python=3.10 -y
conda activate vesuvius

# 4. 安装 PyTorch (CUDA 13.0 → 使用 CUDA 12.1 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. 安装依赖
pip install -r requirements.txt

# 6. 配置 Kaggle API
mkdir -p ~/.kaggle
# 上传 kaggle.json 到 ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

检查清单:
- [ ] Conda 环境创建成功
- [ ] PyTorch 安装成功
- [ ] 依赖包安装成功
- [ ] Kaggle API 配置成功

### Step 4: 环境验证 (5分钟)

```bash
# 运行环境检查脚本
python check_env.py
```

检查清单:
- [ ] Python 版本 >= 3.10
- [ ] PyTorch 可用
- [ ] CUDA 可用
- [ ] GPU: RTX 5090 32GB
- [ ] 所有依赖包已安装
- [ ] Kaggle API 可用

### Step 5: 数据下载 (30-60分钟)

```bash
# 下载比赛数据
python download_data.py
```

检查清单:
- [ ] 数据下载成功
- [ ] 数据解压成功
- [ ] 数据目录结构正确

**预计数据大小**: 
- 原始数据: ~20-50GB
- 预处理后: ~10-30GB

**如果数据盘不足**:
- [ ] 在 AutoDL 控制台扩容数据盘
- [ ] 建议扩容到 200GB

---

## 🎯 训练启动

### 快速启动 (推荐)

```bash
# 使用启动脚本
bash start_training.sh
```

### 手动启动

```bash
# 激活环境
conda activate vesuvius

# 开始训练
python train.py --config configs/autodl_486.yaml
```

检查清单:
- [ ] 训练成功启动
- [ ] GPU 利用率 > 80%
- [ ] 无错误信息
- [ ] WandB 记录正常

---

## 📊 训练监控

### WandB 监控

```bash
# 登录 WandB
wandb login
# 输入 API Key

# 访问仪表板
https://wandb.ai/your-username/vesuvius-challenge
```

检查清单:
- [ ] WandB 登录成功
- [ ] 训练指标正常记录
- [ ] Loss 下降
- [ ] Dice Score 上升

### GPU 监控

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用 gpustat
pip install gpustat
gpustat -i 1
```

检查清单:
- [ ] GPU 利用率 > 80%
- [ ] 显存使用 < 30GB
- [ ] 温度正常 (< 85°C)

---

## 💾 检查点管理

### 自动保存

训练会自动保存:
- `models/checkpoints/best_model.pth` - 最佳模型
- `models/checkpoints/checkpoint_epoch_N.pth` - 每 5 个 epoch

检查清单:
- [ ] 检查点正常保存
- [ ] 磁盘空间充足

### 手动备份（重要！）

```bash
# 定期备份到本地
scp -P [端口] root@[IP]:/root/projects/vesuvius-challenge/models/checkpoints/best_model.pth ./

# 或使用 AutoDL 文件下载功能
```

检查清单:
- [ ] 每天备份一次最佳模型
- [ ] 本地保存检查点

---

## 🔧 常见问题处理

### 问题 1: 显存不足 (OOM)

**症状**: RuntimeError: CUDA out of memory

**解决方案**:
```yaml
# 修改 configs/autodl_486.yaml
training:
  batch_size: 2  # 从 3 减到 2
  patch_size: [64, 64, 64]  # 从 80 减到 64
```

### 问题 2: 数据加载慢

**症状**: GPU 利用率低 (< 50%)

**解决方案**:
```yaml
training:
  num_workers: 12  # 增加到 12
  prefetch_factor: 6  # 增加预加载
```

### 问题 3: 训练中断

**症状**: 连接断开或进程终止

**解决方案**:
```bash
# 使用 screen 或 tmux
screen -S training
python train.py --config configs/autodl_486.yaml

# 断开: Ctrl+A, D
# 重连: screen -r training

# 或从检查点恢复
python train.py --resume models/checkpoints/checkpoint_epoch_20.pth
```

---

## 📈 训练进度追踪

### 预期时间线

| 阶段 | 时间 | 费用 | 检查点 |
|------|------|------|--------|
| 环境配置 | 30分钟 | 1.5元 | 环境检查通过 |
| 数据下载 | 1小时 | 3元 | 数据准备完成 |
| 快速验证 | 2小时 | 6元 | 模型可训练 |
| 基线训练 | 42小时 | 127元 | Dice > 0.7 |
| **总计** | **45.5小时** | **137.5元** | **首次提交** |

### 每日检查

**Day 1**:
- [ ] 环境配置完成
- [ ] 数据下载完成
- [ ] 训练启动成功
- [ ] 训练 5-10 epochs

**Day 2-3**:
- [ ] 训练持续进行
- [ ] Loss 稳定下降
- [ ] Dice Score > 0.6
- [ ] 备份检查点

**Day 4-5**:
- [ ] 训练接近完成
- [ ] Dice Score > 0.7
- [ ] 准备推理
- [ ] 生成提交

---

## 🎯 成功标准

### 训练成功

- [ ] 50 epochs 完成
- [ ] Val Dice Score > 0.7
- [ ] 无 OOM 错误
- [ ] 检查点保存完整

### 提交成功

- [ ] 推理完成
- [ ] submission.csv 生成
- [ ] 提交到 Kaggle
- [ ] 获得 Public Score

---

## 💰 成本控制

### 实时费用追踪

```bash
# 计算当前费用
开始时间: [记录]
当前时间: [记录]
运行时长: [计算]
当前费用: 时长 × 3.03元
```

### 省钱技巧

- [ ] 训练完成立即释放实例
- [ ] 使用检查点暂停/恢复
- [ ] 数据预处理在 Kaggle 完成
- [ ] 推理可以用更便宜的 GPU

---

## 📞 紧急联系

### 遇到问题时

1. **查看文档**
   - README.md
   - QUICK_START.md
   - COMPETITION_PLAN.md

2. **查看日志**
   ```bash
   tail -f logs/training.log
   ```

3. **联系支持**
   - AutoDL 客服
   - Kaggle Discussion
   - 找我帮助！

---

## ✅ 最终检查

训练完成后:

- [ ] 最佳模型已保存
- [ ] 检查点已备份到本地
- [ ] WandB 记录完整
- [ ] 准备推理和提交
- [ ] 释放 AutoDL 实例（如果暂时不用）

---

**祝训练顺利！** 🚀

**有任何问题随时联系我！** 💪
