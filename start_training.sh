#!/bin/bash

echo "=========================================="
echo "Vesuvius Challenge - AutoDL 486机"
echo "RTX 5090 (32GB) 训练启动"
echo "=========================================="
echo ""

# 激活环境
echo "激活 Conda 环境..."
source activate vesuvius

# 检查 GPU
echo ""
echo "检查 GPU 状态..."
nvidia-smi
echo ""

# 检查 CUDA
echo "CUDA 版本:"
nvcc --version
echo ""

# 检查 PyTorch
echo "PyTorch 版本:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# 检查数据
echo "检查数据目录..."
if [ -d "data/processed/train" ]; then
    echo "✓ 训练数据存在"
else
    echo "✗ 训练数据不存在，请先运行 python download_data.py"
    exit 1
fi

if [ -d "data/processed/val" ]; then
    echo "✓ 验证数据存在"
else
    echo "✗ 验证数据不存在"
    exit 1
fi

echo ""
echo "=========================================="
echo "开始训练..."
echo "=========================================="
echo ""
echo "配置: configs/autodl_486.yaml"
echo "Batch Size: 3"
echo "Patch Size: 80×80×80"
echo "Epochs: 50"
echo ""

# 训练
python train.py --config configs/autodl_486.yaml

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "检查点保存在: models/checkpoints/"
echo "最佳模型: models/checkpoints/best_model.pth"
echo ""
