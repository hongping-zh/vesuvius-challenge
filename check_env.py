"""
环境检查脚本

检查 AutoDL 环境是否配置正确
"""

import sys
import subprocess


def check_python():
    """检查 Python 版本"""
    print("=" * 60)
    print("检查 Python 版本")
    print("=" * 60)
    version = sys.version
    print(f"Python 版本: {version}")
    
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 8:
        print("✓ Python 版本符合要求 (>= 3.8)")
        return True
    else:
        print("✗ Python 版本过低，需要 >= 3.8")
        return False


def check_torch():
    """检查 PyTorch"""
    print("\n" + "=" * 60)
    print("检查 PyTorch")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
            
            # 检查显存
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"显存大小: {total_memory:.2f} GB")
            
            if total_memory >= 30:
                print("✓ 显存充足 (>= 30GB)")
                return True
            else:
                print("⚠ 显存可能不足")
                return True
        else:
            print("✗ CUDA 不可用")
            return False
            
    except ImportError:
        print("✗ PyTorch 未安装")
        return False


def check_dependencies():
    """检查依赖包"""
    print("\n" + "=" * 60)
    print("检查依赖包")
    print("=" * 60)
    
    packages = [
        'numpy',
        'pandas',
        'opencv-python',
        'albumentations',
        'segmentation-models-pytorch',
        'wandb',
        'tqdm',
        'pyyaml'
    ]
    
    all_installed = True
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} 未安装")
            all_installed = False
    
    return all_installed


def check_gpu_memory():
    """检查 GPU 显存使用"""
    print("\n" + "=" * 60)
    print("检查 GPU 显存")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            # 分配测试张量
            device = torch.device('cuda')
            
            # 测试不同大小的张量
            sizes = [
                (1, 1, 64, 64, 64),
                (2, 1, 64, 64, 64),
                (3, 1, 64, 64, 64),
                (1, 1, 80, 80, 80),
                (2, 1, 80, 80, 80),
                (3, 1, 80, 80, 80),
            ]
            
            for size in sizes:
                try:
                    x = torch.randn(*size, device=device)
                    memory_used = torch.cuda.memory_allocated() / 1e9
                    print(f"✓ {size}: {memory_used:.2f} GB")
                    del x
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    print(f"✗ {size}: 显存不足")
                    break
            
            return True
        else:
            print("✗ CUDA 不可用")
            return False
            
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


def check_data():
    """检查数据目录"""
    print("\n" + "=" * 60)
    print("检查数据目录")
    print("=" * 60)
    
    from pathlib import Path
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("✗ data/ 目录不存在")
        return False
    
    # 检查子目录
    subdirs = ['raw', 'processed', 'processed/train', 'processed/val']
    
    for subdir in subdirs:
        path = data_dir / subdir
        if path.exists():
            print(f"✓ {subdir}/")
        else:
            print(f"✗ {subdir}/ 不存在")
    
    return True


def check_kaggle_api():
    """检查 Kaggle API"""
    print("\n" + "=" * 60)
    print("检查 Kaggle API")
    print("=" * 60)
    
    from pathlib import Path
    
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if kaggle_json.exists():
        print("✓ kaggle.json 存在")
        
        # 测试 API
        try:
            result = subprocess.run(
                ['kaggle', 'competitions', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print("✓ Kaggle API 可用")
                return True
            else:
                print("✗ Kaggle API 错误")
                return False
                
        except Exception as e:
            print(f"✗ Kaggle API 测试失败: {e}")
            return False
    else:
        print("✗ kaggle.json 不存在")
        print("请将 kaggle.json 放到 ~/.kaggle/")
        return False


def main():
    """主函数"""
    print("\n")
    print("🔍 Vesuvius Challenge 环境检查")
    print("\n")
    
    results = {
        'Python': check_python(),
        'PyTorch': check_torch(),
        '依赖包': check_dependencies(),
        'GPU 显存': check_gpu_memory(),
        '数据目录': check_data(),
        'Kaggle API': check_kaggle_api()
    }
    
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    
    for name, result in results.items():
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 环境检查通过！可以开始训练！")
    else:
        print("❌ 环境检查未通过，请修复上述问题")
    print("=" * 60)
    print("\n")
    
    if all_passed:
        print("下一步:")
        print("1. 下载数据: python download_data.py")
        print("2. 开始训练: python train.py --config configs/autodl_486.yaml")
        print("或使用: bash start_training.sh")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
