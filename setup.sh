#!/bin/bash

echo "========================================="
echo "Vesuvius Challenge - 环境配置"
echo "========================================="
echo ""

# 创建 conda 环境
echo "创建 Python 环境..."
conda create -n vesuvius python=3.10 -y
source activate vesuvius

# 安装 PyTorch (CUDA 12.1)
echo "安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装核心依赖
echo "安装核心依赖..."
pip install -r requirements.txt

# 配置 Kaggle API
echo "配置 Kaggle API..."
mkdir -p ~/.kaggle
echo "请将 kaggle.json 放到 ~/.kaggle/ 目录"

# 创建必要目录
echo "创建项目目录..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p outputs
mkdir -p logs

echo ""
echo "========================================="
echo "环境配置完成！"
echo "========================================="
echo ""
echo "下一步："
echo "1. 将 kaggle.json 放到 ~/.kaggle/"
echo "2. 运行: python download_data.py"
echo "3. 运行: python train.py"
