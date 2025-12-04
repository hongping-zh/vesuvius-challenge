# 320³ 推理操作检查清单

快速参考，执行前逐项勾选。

---

## 📋 准备阶段

### Kaggle 端
- [ ] 登录 Kaggle
- [ ] 创建新 Notebook 或使用现有 Notebook
- [ ] 确保 Internet 已打开
- [ ] 运行以下代码下载测试数据：

```python
import tifffile as tiff
import numpy as np
from pathlib import Path

test_img_path = "/kaggle/input/vesuvius-challenge-surface-detection/test_images/1407735.tif"
test_img = tiff.imread(test_img_path)
np.save("/kaggle/working/test_volume_320.npy", test_img)
print(f"✓ Saved: {Path('/kaggle/working/test_volume_320.npy').stat().st_size / 1e6:.2f} MB")
```

- [ ] 从 Notebook Output 下载 `test_volume_320.npy` 到本地
- [ ] 验证文件大小约 32 MB

---

### AutoDL 端
- [ ] SSH 登录 AutoDL
- [ ] 检查端口号（从控制台复制）
- [ ] 验证项目路径：`/root/autodl-tmp/vesuvius-challenge/`
- [ ] 确保 `torch_env` 环境可用：`conda env list`
- [ ] 确保模型权重存在：`ls -lh models/checkpoints_dynunet_realdata_8epoch/best_model.pth`

---

### 本地端
- [ ] PowerShell 或终端已打开
- [ ] `test_volume_320.npy` 已下载
- [ ] SCP 命令已准备好（替换端口号）

---

## 🚀 执行阶段

### Step 1: 上传测试数据到 AutoDL

在本地 PowerShell 运行（替换 `<port>` 为你的端口号）：

```powershell
scp -P <port> test_volume_320.npy root@connect.westb.seetacloud.com:/root/autodl-tmp/vesuvius-challenge/data/processed/test/
```

验证：
- [ ] 输入密码
- [ ] 上传进度显示
- [ ] 上传完成（100%）

---

### Step 2: 验证上传成功

SSH 登录 AutoDL，运行：

```bash
cd /root/autodl-tmp/vesuvius-challenge
ls -lh data/processed/test/test_volume_320.npy
```

检查：
- [ ] 文件存在
- [ ] 文件大小约 32 MB

---

### Step 3: 上传新的推理脚本（可选）

如果想用我准备的新脚本 `run_inference_autodl_320.py`：

```powershell
scp -P <port> run_inference_autodl_320.py root@connect.westb.seetacloud.com:/root/autodl-tmp/vesuvius-challenge/
```

或者直接在 AutoDL 上修改原脚本：

```bash
cd /root/autodl-tmp/vesuvius-challenge
nano run_inference_autodl.py
# 修改第 33 行，改为：
# TEST_VOLUME_PATH = PROJECT_ROOT / "data" / "processed" / "test" / "test_volume_320.npy"
# Ctrl+O 保存，Ctrl+X 退出
```

验证：
- [ ] 脚本已修改或上传
- [ ] `TEST_VOLUME_PATH` 指向 `test_volume_320.npy`

---

### Step 4: 运行推理

SSH 登录 AutoDL，运行：

```bash
conda activate torch_env
cd /root/autodl-tmp/vesuvius-challenge
python run_inference_autodl_320.py  # 或 run_inference_autodl.py（如果修改了原脚本）
```

监控输出：
- [ ] 显示 "Using device: cuda"
- [ ] 显示 "Shape: (320, 320, 320)"
- [ ] 显示 "✓ Correct volume size: (320, 320, 320)"
- [ ] Inference 进度条正常运行
- [ ] 显示 "Submission TIF generated"
- [ ] 显示 "Shape: (320, 320, 320)"（最终输出）
- [ ] 没有报错

预计时间：15-30 分钟（取决于 GPU）

---

### Step 5: 下载结果

在本地 PowerShell 运行：

```powershell
scp -P <port> root@connect.westb.seetacloud.com:/root/autodl-tmp/vesuvius-challenge/prediction.tif ./prediction_320.tif
```

验证：
- [ ] 下载完成
- [ ] 文件大小约 32 MB

---

### Step 6: 本地验证

在本地运行 Python：

```python
import tifffile as tiff
import numpy as np

pred = tiff.imread("prediction_320.tif")
print(f"Shape: {pred.shape}")
print(f"Dtype: {pred.dtype}")
print(f"Unique: {np.unique(pred)}")
print(f"Positive: {pred.mean():.6f}")
```

检查：
- [ ] Shape 是 `(320, 320, 320)`
- [ ] Dtype 是 `uint8`
- [ ] Unique 是 `[0 1]` 或 `[0]` 或 `[1]`
- [ ] Positive ratio 在合理范围（0.001 - 0.1）

---

## 📤 提交阶段

### Step 7: 更新 Kaggle Dataset

1. 访问你的 Dataset（或创建新的）
   - [ ] 打开 https://www.kaggle.com/datasets
   - [ ] 找到 `vesuvius-dynunet-prediction-tif` 或创建新 Dataset

2. 创建新版本
   - [ ] 点击 **New Version**
   - [ ] 删除旧的 `prediction.tif`（如果有）
   - [ ] 上传新的 `prediction_320.tif`
   - [ ] 重命名为 `prediction.tif`
   - [ ] 版本说明填写：`Native 320^3 prediction without resize`
   - [ ] 点击 **Create**

3. 等待处理
   - [ ] Dataset 状态变为 "Complete"
   - [ ] 可以在 Notebook 中访问

---

### Step 8: 更新 Kaggle Notebook

1. 打开提交 Notebook
   - [ ] Internet 已打开
   - [ ] 链接新版本的 Dataset

2. 简化代码（不需要 resize 了）
   ```python
   import zipfile
   from pathlib import Path
   
   src = Path("/kaggle/input/vesuvius-dynunet-prediction-tif/prediction.tif")
   dst = Path("/kaggle/working/prediction.tif")
   
   dst.write_bytes(src.read_bytes())
   print("✓ TIF copied")
   
   zip_path = Path("/kaggle/working/submission.zip")
   with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
       zf.write(dst, arcname="prediction.tif")
   
   print("✓ Submission zip created")
   ```

3. 运行测试
   - [ ] Internet 打开，Save & Run All
   - [ ] 验证输出正确
   - [ ] 删除安装 imagecodecs 的命令（如果有）

4. 最终提交
   - [ ] Internet 关闭
   - [ ] Save & Run All
   - [ ] 等待运行完成
   - [ ] 验证 `submission.zip` 已生成

---

### Step 9: 提交到竞赛

1. 提交 Notebook
   - [ ] 点击右上角 **Submit**
   - [ ] 选择最新版本
   - [ ] 填写描述：`Native 320^3 prediction without resize`
   - [ ] 点击 **Submit**

2. 等待评分
   - [ ] 状态显示 "Scoring"
   - [ ] 等待 5-10 分钟
   - [ ] 检查是否成功得分

3. 检查结果
   - [ ] 没有 Scoring Error
   - [ ] 得分显示（无论分数高低，只要有分数就成功）

---

## ✅ 完成

恭喜！你已经成功在 AutoDL 上用真实的 320³ 测试数据完成推理并提交。

### 后续优化建议

如果分数不理想，可以尝试：

- [ ] 调整 `overlap`：0.5 → 0.75（更高精度）
- [ ] 调整 `threshold`：0.3 → 0.2 或 0.4（试验最佳值）
- [ ] 训练更多 epoch
- [ ] 使用数据增强
- [ ] 尝试不同的模型架构

---

## 🆘 遇到问题？

参考：
- 详细指南：`AUTODL_320_INFERENCE_GUIDE.md`
- 原推理脚本：`run_inference_autodl.py`
- 新推理脚本：`run_inference_autodl_320.py`

或随时问我！
