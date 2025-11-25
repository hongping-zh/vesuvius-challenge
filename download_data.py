"""
下载 Vesuvius Challenge 数据

使用 Kaggle API 下载比赛数据
"""

import os
import subprocess
from pathlib import Path
import zipfile


def download_competition_data():
    """下载比赛数据"""
    print("=" * 60)
    print("下载 Vesuvius Challenge 数据")
    print("=" * 60)
    print()
    
    # 检查 Kaggle API 配置
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌ 错误: 未找到 kaggle.json")
        print("请执行以下步骤:")
        print("1. 访问 https://www.kaggle.com/settings")
        print("2. 点击 'Create New API Token'")
        print("3. 将下载的 kaggle.json 放到 ~/.kaggle/")
        print("4. 运行: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # 创建数据目录
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载比赛数据
    print("📥 下载比赛数据...")
    try:
        subprocess.run([
            "kaggle", "competitions", "download",
            "-c", "vesuvius-challenge-surface-detection",
            "-p", str(data_dir)
        ], check=True)
        
        print("✅ 数据下载完成")
        
        # 解压数据
        print("\n📦 解压数据...")
        for zip_file in data_dir.glob("*.zip"):
            print(f"  解压: {zip_file.name}")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # 删除 zip 文件以节省空间
            zip_file.unlink()
        
        print("✅ 数据解压完成")
        
        # 显示数据结构
        print("\n📁 数据结构:")
        for item in data_dir.rglob("*"):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.relative_to(data_dir)} ({size_mb:.2f} MB)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def download_sample_data():
    """下载示例数据（用于测试）"""
    print("\n📥 下载示例数据...")
    
    # 这里可以添加下载小样本数据的代码
    # 用于快速测试和开发
    
    print("✅ 示例数据准备完成")


if __name__ == "__main__":
    success = download_competition_data()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ 数据准备完成！")
        print("=" * 60)
        print("\n下一步:")
        print("1. 运行: python preprocess.py")
        print("2. 运行: python train.py")
    else:
        print("\n" + "=" * 60)
        print("❌ 数据下载失败")
        print("=" * 60)
