#!/bin/bash
# AutoDL 快速设置脚本

echo "============================================================"
echo "Vesuvius Challenge - AutoDL 快速设置"
echo "============================================================"

# 1. 安装依赖
echo ""
echo "📦 安装依赖..."
pip install monai[all]==1.3.2
pip install connected-components-3d
pip install albumentations
pip install tifffile
pip install zarr
pip install scikit-image

# 验证安装
echo ""
echo "✅ 验证安装..."
python -c "import monai; print(f'MONAI {monai.__version__}')"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 2. 测试 DynUNet
echo ""
echo "🧪 测试 DynUNet..."
python test_dynunet.py

# 3. 创建必要目录
echo ""
echo "📁 创建目录..."
mkdir -p data/raw
mkdir -p data/processed/train
mkdir -p data/processed/val
mkdir -p models/checkpoints_dynunet_small
mkdir -p logs

echo ""
echo "============================================================"
echo "✅ 设置完成！"
echo "============================================================"
echo ""
echo "下一步:"
echo "1. 配置 Kaggle API: mkdir -p ~/.kaggle && vim ~/.kaggle/kaggle.json"
echo "2. 下载数据: python download_data.py"
echo "3. 开始训练: python train.py --config configs/autodl_dynunet_small.yaml"
echo ""
echo "使用 tmux 保持会话:"
echo "  tmux new -s vesuvius"
echo "  python train.py --config configs/autodl_dynunet_small.yaml"
echo "  Ctrl+B, D (分离会话)"
echo "  tmux attach -t vesuvius (重新连接)"
echo ""
