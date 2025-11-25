#!/bin/bash

# 一键部署脚本：用于在 AutoDL 上快速准备环境并自检
# 使用方法（在 AutoDL 远程主机上）：
#   bash deploy_autodl.sh

set -e

REPO_URL="https://github.com/hongping-zh/vesuvius-challenge.git"
REPO_NAME="vesuvius-challenge"

if [ ! -d "$REPO_NAME" ]; then
  echo "Cloning repository from $REPO_URL ..."
  git clone "$REPO_URL"
fi

cd "$REPO_NAME"

echo "Running autodl_setup.sh ..."
if [ -x "autodl_setup.sh" ]; then
  bash autodl_setup.sh
else
  echo "❌ 找不到 autodl_setup.sh，请确认脚本已提交到仓库。"
  exit 1
fi

echo "Running test_optimizations.py ..."
python test_optimizations.py

echo "✅ AutoDL 部署完成，可以按 READY_FOR_TRAINING.md 中的三步走进行训练。"
