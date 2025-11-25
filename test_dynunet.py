"""
测试 DynUNet 模型
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_dynunet_import():
    """测试 DynUNet 导入"""
    print("\n" + "="*60)
    print("测试 1: 导入 DynUNet")
    print("="*60)
    
    try:
        from models.dynunet import VesuviusDynUNet
        print("✅ DynUNet 导入成功")
        return True
    except ImportError as e:
        print(f"❌ DynUNet 导入失败: {e}")
        return False


def test_dynunet_creation():
    """测试 DynUNet 创建"""
    print("\n" + "="*60)
    print("测试 2: 创建 DynUNet 模型")
    print("="*60)
    
    try:
        from models.dynunet import VesuviusDynUNet
        
        model = VesuviusDynUNet(
            in_channels=1,
            base_num_features=64,
            num_classes=1,
            deep_supervision=True
        )
        
        print("✅ DynUNet 创建成功")
        return True
    except Exception as e:
        print(f"❌ DynUNet 创建失败: {e}")
        return False


def test_dynunet_forward():
    """测试 DynUNet 前向传播"""
    print("\n" + "="*60)
    print("测试 3: DynUNet 前向传播")
    print("="*60)
    
    try:
        from models.dynunet import VesuviusDynUNet
        
        model = VesuviusDynUNet(
            in_channels=1,
            base_num_features=32,  # 小一点，快速测试
            num_classes=1,
            deep_supervision=True
        )
        
        # 测试输入
        x = torch.randn(1, 1, 96, 96, 96)
        print(f"\n输入形状: {x.shape}")
        
        # 训练模式
        model.train()
        with torch.no_grad():
            out_train = model(x)
        
        if isinstance(out_train, list):
            print(f"\n训练模式输出（深度监督）:")
            for i, o in enumerate(out_train):
                print(f"  输出 {i}: {o.shape}")
        else:
            print(f"\n训练模式输出: {out_train.shape}")
        
        # 推理模式
        model.eval()
        with torch.no_grad():
            out_eval = model(x)
        
        print(f"\n推理模式输出: {out_eval.shape}")
        
        print("\n✅ 前向传播测试通过")
        return True
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """测试配置文件加载"""
    print("\n" + "="*60)
    print("测试 4: 加载 DynUNet 配置")
    print("="*60)
    
    try:
        import yaml
        
        config_path = Path(__file__).parent / 'configs' / 'autodl_dynunet_small.yaml'
        
        if not config_path.exists():
            print(f"❌ 配置文件不存在: {config_path}")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"\n配置文件: {config_path.name}")
        print(f"模型类型: {config['model']['type']}")
        print(f"输入通道: {config['model']['in_channels']}")
        print(f"基础特征: {config['model']['base_num_features']}")
        print(f"深度监督: {config['model']['deep_supervision']}")
        print(f"Patch Size: {config['data']['patch_size']}")
        print(f"Batch Size: {config['training']['batch_size']}")
        print(f"Epochs: {config['training']['epochs']}")
        
        print("\n✅ 配置文件加载成功")
        return True
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("DynUNet 完整测试")
    print("="*60)
    
    results = []
    
    # 测试 1: 导入
    results.append(("导入测试", test_dynunet_import()))
    
    # 测试 2: 创建
    results.append(("创建测试", test_dynunet_creation()))
    
    # 测试 3: 前向传播
    results.append(("前向传播测试", test_dynunet_forward()))
    
    # 测试 4: 配置加载
    results.append(("配置加载测试", test_config_loading()))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(r for _, r in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！")
        print("\n下一步:")
        print("1. 安装 MONAI: pip install monai[all]==1.3.2")
        print("2. 运行推理测试: python inference_notebook.py")
        print("3. 在 AutoDL 上训练: python train.py --config configs/autodl_dynunet_small.yaml")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
    
    print("="*60)


if __name__ == "__main__":
    main()
