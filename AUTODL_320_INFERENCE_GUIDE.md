# AutoDL 320³ 真实测试数据推理方案

## 📋 目标

在 AutoDL 上使用 Kaggle 真实的 **320×320×320** 测试数据进行推理，生成正确尺寸的 `prediction.tif`，避免 resize 导致的精度损失。

---

## 🔍 问题分析

### 当前状态
- ❌ AutoDL 上的测试数据是 **128³** (来自训练集的一个样本)
- ❌ Kaggle 真实测试数据是 **320³**
- ❌ 用 128³ 推理 → resize 到 320³ → 精度损失

### 目标状态
- ✅ 直接在 320³ 数据上推理
- ✅ 输出原生 320³ 预测，无需 resize
- ✅ 最大化保留模型精度

---

## 📦 方案一：下载真实测试数据到 AutoDL（推荐）

### Step 1: 在 Kaggle 上下载测试数据

#### 1.1 创建下载 Notebook

在 Kaggle 创建新 Notebook，运行以下代码：

```python
import tifffile as tiff
import numpy as np
from pathlib import Path

# 读取测试图像
test_img_path = "/kaggle/input/vesuvius-challenge-surface-detection/test_images/1407735.tif"
test_img = tiff.imread(test_img_path)

print(f"Test image shape: {test_img.shape}")  # (320, 320, 320)
print(f"Test image dtype: {test_img.dtype}")  # uint8

# 保存为 .npy 格式（方便 AutoDL 加载）
output_path = Path("/kaggle/working/test_volume_320.npy")
np.save(output_path, test_img)
print(f"Saved to: {output_path}")
print(f"File size: {output_path.stat().st_size / 1e6:.2f} MB")
```

#### 1.2 下载文件到本地

运行完成后，从 Notebook 的 Output 中下载 `test_volume_320.npy`（约 32 MB）。

---

### Step 2: 上传到 AutoDL

#### 2.1 使用 SCP 上传

在本地 PowerShell 运行：

```powershell
# 假设你的 AutoDL SSH 端口是 43898（从控制台查看）
# 替换 <your-port> 为你的实际端口号

scp -P <your-port> test_volume_320.npy root@connect.westb.seetacloud.com:/root/autodl-tmp/vesuvius-challenge/data/processed/test/
```

输入密码后等待上传完成。

#### 2.2 验证上传成功

SSH 登录 AutoDL，运行：

```bash
cd /root/autodl-tmp/vesuvius-challenge
ls -lh data/processed/test/test_volume_320.npy
```

应该看到文件大小约 32 MB。

---

### Step 3: 修改推理脚本

在 AutoDL 上编辑 `run_inference_autodl.py`：

#### 3.1 修改测试数据路径

找到第 33 行：

```python
# 原代码：
TEST_VOLUME_PATH = PROJECT_ROOT / "data" / "processed" / "test" / "volume.npy"

# 改为：
TEST_VOLUME_PATH = PROJECT_ROOT / "data" / "processed" / "test" / "test_volume_320.npy"
```

#### 3.2 修改 patch size（如果需要）

如果你的模型训练时用的是 128³ patch，保持不变：

```python
patch_size = tuple(config.get("data", {}).get("patch_size", [128, 128, 128]))
```

模型会用滑动窗口在 320³ 数据上推理。

#### 3.3 调整 overlap（可选，提高精度）

在 `main()` 函数中，找到 `sliding_window_inference` 调用（约 267 行）：

```python
preds = sliding_window_inference(
    model=model,
    volume=volume,
    patch_size=patch_size,
    overlap=0.5,  # 可以改为 0.75 提高精度，但推理时间更长
    batch_size=2,
    device=str(device),
    in_channels=config["model"].get("in_channels", 1),
)
```

**overlap 建议**：
- `0.5`：默认值，平衡速度和精度
- `0.75`：更高精度，推理时间约 2 倍
- `0.25`：更快速度，精度略低

---

### Step 4: 运行推理

在 AutoDL 的 `torch_env` 环境中运行：

```bash
conda activate torch_env
cd /root/autodl-tmp/vesuvius-challenge
python run_inference_autodl.py
```

#### 预期输出

```
============================================================
AutoDL DynUNet One-click Inference
============================================================

Using device: cuda
Patch size: (128, 128, 128)
Loading weights: .../best_model.pth
Loading test volume: .../test_volume_320.npy
  Shape: (320, 320, 320)
  Range: [0.0000, 255.0000]
Total patches: 125  # 320³ 用 128³ patch，overlap=0.5 时约 125 个 patch
Inference: 100%|████████████████| 63/63 [XX:XX<00:00]

Inference completed, prediction range: [0.0123, 0.9876]
Probability map saved to: predictions_dynunet.npy

Post-processing:
  Threshold: 0.3
  Prediction range: [0.0123, 0.9876]
  Positive ratio after threshold: 0.034567

Submission TIF generated: prediction.tif
  Shape: (320, 320, 320)  # ✅ 正确尺寸！
  Unique values: [0 1]
  Positive ratio: 0.034567

============================================================
One-click inference completed!
============================================================
Total time: 0h 15m  # 时间会更长，因为数据更大
```

---

### Step 5: 下载结果

#### 5.1 从 AutoDL 下载

在本地 PowerShell 运行：

```powershell
scp -P <your-port> root@connect.westb.seetacloud.com:/root/autodl-tmp/vesuvius-challenge/prediction.tif ./prediction_320.tif
```

#### 5.2 验证尺寸

在本地运行 Python 验证：

```python
import tifffile as tiff
import numpy as np

pred = tiff.imread("prediction_320.tif")
print(f"Shape: {pred.shape}")  # 应该是 (320, 320, 320)
print(f"Dtype: {pred.dtype}")  # 应该是 uint8
print(f"Unique values: {np.unique(pred)}")  # 应该是 [0, 1]
```

---

### Step 6: 上传到 Kaggle Dataset

#### 6.1 更新 Kaggle Dataset

如果之前的 Dataset 已存在，创建新版本：

1. 访问 https://www.kaggle.com/datasets/yourname/vesuvius-dynunet-prediction-tif
2. 点击 **New Version**
3. 上传新的 `prediction_320.tif`（替换旧文件）
4. 添加版本说明：`Native 320^3 prediction without resize`
5. 点击 **Create**

#### 6.2 更新 Kaggle Notebook

Notebook 代码可以简化（不需要 resize 了）：

```python
import zipfile
from pathlib import Path

# 直接复制，无需 resize
src = Path("/kaggle/input/vesuvius-dynunet-prediction-tif/prediction.tif")
dst = Path("/kaggle/working/prediction.tif")

dst.write_bytes(src.read_bytes())
print("TIF copied to:", dst)
print("TIF size:", dst.stat().st_size, "bytes")

# 创建 submission.zip
zip_path = Path("/kaggle/working/submission.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(dst, arcname="prediction.tif")

print("Submission zip created:", zip_path)
with zipfile.ZipFile(zip_path, "r") as zf:
    print("Files in zip:", zf.namelist())
```

---

## 📦 方案二：直接在 Kaggle Notebook 中推理（备选）

如果 AutoDL 数据传输麻烦，可以考虑直接在 Kaggle Notebook 中运行推理。

### 优点
- ✅ 测试数据已在 Kaggle，无需下载/上传
- ✅ 无需在 AutoDL 和本地之间传输文件
- ✅ GPU 免费（每周 30 小时）

### 缺点
- ❌ 需要上传模型权重到 Kaggle Dataset
- ❌ 需要在 Notebook 中重新实现推理代码
- ❌ Kaggle GPU 性能可能略低于 AutoDL（取决于实例）

### 实现步骤（简要）

1. **上传模型到 Kaggle Dataset**
   - 从 AutoDL 下载 `best_model.pth`
   - 创建 Kaggle Dataset 上传

2. **创建推理 Notebook**
   - 链接模型 Dataset 和测试数据 Dataset
   - 复制 `run_inference_autodl.py` 的推理逻辑
   - 直接在 Kaggle GPU 上运行

3. **生成提交**
   - 直接保存 `prediction.tif` 和 `submission.zip`

（如需详细步骤，告诉我，我可以展开）

---

## 🎯 推荐流程总结

### 最简方案（推荐新手）

1. ✅ 在 Kaggle 下载 `test_volume_320.npy`
2. ✅ SCP 上传到 AutoDL
3. ✅ 修改 `run_inference_autodl.py` 的路径
4. ✅ 运行推理，得到 320³ 预测
5. ✅ 下载并提交

### 最优方案（推荐熟练用户）

直接在 Kaggle Notebook 中推理（方案二），避免文件来回传输。

---

## ⏱️ 时间估算

| 步骤 | 时间 |
|-----|------|
| Kaggle 下载测试数据 | 2 分钟 |
| SCP 上传到 AutoDL | 5-10 分钟（取决于网速）|
| 修改脚本 | 2 分钟 |
| AutoDL 推理（320³） | 15-30 分钟（取决于 GPU）|
| 下载结果 | 2 分钟 |
| 上传 Kaggle 提交 | 5 分钟 |
| **总计** | **约 30-50 分钟** |

---

## 🔧 可能的问题和解决

### Q1: SCP 上传很慢怎么办？

**方案 A**：使用 rsync（如果支持）
```bash
rsync -avz -e "ssh -p <port>" test_volume_320.npy root@connect.westb.seetacloud.com:/root/autodl-tmp/vesuvius-challenge/data/processed/test/
```

**方案 B**：压缩后上传
```bash
# 本地压缩
gzip test_volume_320.npy

# 上传 .gz 文件（更小）
scp -P <port> test_volume_320.npy.gz root@...

# AutoDL 上解压
gunzip test_volume_320.npy.gz
```

**方案 C**：改用方案二（直接在 Kaggle 推理）

---

### Q2: AutoDL GPU 内存不够怎么办？

调整 `batch_size` 和 `patch_size`：

```python
# 在 main() 函数中
preds = sliding_window_inference(
    model=model,
    volume=volume,
    patch_size=(96, 96, 96),  # 减小 patch size
    overlap=0.5,
    batch_size=1,  # 减小 batch size
    device=str(device),
    in_channels=config["model"].get("in_channels", 1),
)
```

---

### Q3: 推理时间太长怎么办？

- 降低 `overlap`：`0.5` → `0.25`
- 增大 `batch_size`（如果 GPU 内存允许）
- 使用更快的 GPU（AutoDL 升级实例）

---

## 📝 下一步行动

### 学习阶段（现在）

1. **阅读本文档**，理解整个流程
2. **准备工具**：
   - 确保本地安装了 Python + tifffile
   - 确保能 SSH 登录 AutoDL
   - 确保能访问 Kaggle

### 执行阶段（明天或准备好后）

1. **Kaggle 下载测试数据**（方案一 Step 1）
2. **上传到 AutoDL**（方案一 Step 2）
3. **修改推理脚本**（方案一 Step 3）
4. **运行推理**（方案一 Step 4）
5. **提交到 Kaggle**（方案一 Step 5-6）

---

## 📚 参考资料

- Kaggle 测试数据路径：`/kaggle/input/vesuvius-challenge-surface-detection/test_images/1407735.tif`
- AutoDL 项目路径：`/root/autodl-tmp/vesuvius-challenge/`
- SCP 使用文档：https://linux.die.net/man/1/scp

---

**祝学习顺利！有任何问题随时问我。** 🚀
