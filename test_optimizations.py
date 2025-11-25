"""
测试所有优化功能
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))


def test_ink_sampling():
    """测试 Ink-only Sampling"""
    print("\n" + "="*60)
    print("测试 1: Ink-only Sampling")
    print("="*60)
    
    try:
        from utils.ink_sampling import InkAwareVesuviusDataset
        print("✅ 导入成功")
        
        # 注意：需要真实数据才能测试
        print("⚠️  需要真实数据才能完整测试")
        print("   数据路径: data/processed/train/")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_multi_channel():
    """测试多通道特征"""
    print("\n" + "="*60)
    print("测试 2: 多通道特征提取")
    print("="*60)
    
    try:
        from utils.multi_channel import extract_multi_channel_features, get_channel_count
        import numpy as np
        
        # 创建测试数据
        volume = np.random.rand(64, 64, 64).astype(np.float32)
        
        # 测试不同通道组合
        test_configs = [
            ['raw'],
            ['raw', 'grad'],
            ['raw', 'grad', 'log'],
        ]
        
        for channels in test_configs:
            features = extract_multi_channel_features(volume, channels)
            expected_channels = get_channel_count(channels)
            
            print(f"\n通道配置: {channels}")
            print(f"  输出形状: {features.shape}")
            print(f"  预期通道: {expected_channels}")
            print(f"  实际通道: {features.shape[0]}")
            
            assert features.shape[0] == expected_channels, "通道数不匹配！"
        
        print("\n✅ 测试通过")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamic_loss():
    """测试动态 Loss 调度"""
    print("\n" + "="*60)
    print("测试 3: 动态 Loss 权重调度")
    print("="*60)
    
    try:
        from utils.dynamic_loss import DynamicLossScheduler
        
        scheduler = DynamicLossScheduler(
            total_epochs=50,
            warmup_epochs=20,
            strategy='two_stage'
        )
        
        # 测试关键 epoch
        test_epochs = [0, 10, 19, 20, 30, 49]
        
        print(f"\n{'Epoch':<8} {'Dice':<8} {'BCE':<8} {'Surface':<10} {'Topology':<10}")
        print("-"*60)
        
        for epoch in test_epochs:
            weights = scheduler.get_weights(epoch)
            print(f"{epoch:<8} {weights['dice']:<8.3f} {weights['bce']:<8.3f} "
                  f"{weights['surface']:<10.3f} {weights['topology']:<10.3f}")
            
            # 验证权重
            if epoch < 20:
                assert weights['surface'] == 0.0, f"Epoch {epoch} surface 应该为 0"
                assert weights['topology'] == 0.0, f"Epoch {epoch} topology 应该为 0"
            else:
                assert weights['surface'] > 0.0, f"Epoch {epoch} surface 应该 > 0"
                assert weights['topology'] > 0.0, f"Epoch {epoch} topology 应该 > 0"
        
        print("\n✅ 测试通过")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """测试优化配置加载"""
    print("\n" + "="*60)
    print("测试 4: 优化配置文件")
    print("="*60)
    
    try:
        import yaml
        
        config_path = Path(__file__).parent / 'configs' / 'autodl_dynunet_optimized.yaml'
        
        if not config_path.exists():
            print(f"❌ 配置文件不存在: {config_path}")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"\n配置文件: {config_path.name}")
        print(f"\n模型配置:")
        print(f"  类型: {config['model']['type']}")
        print(f"  输入通道: {config['model']['in_channels']}")
        print(f"  基础特征: {config['model']['base_num_features']}")
        
        print(f"\n数据配置:")
        print(f"  数据集类型: {config['data']['dataset_type']}")
        print(f"  通道: {config['data']['channels']}")
        print(f"  Patch Size: {config['data']['patch_size']}")
        print(f"  Positive Ratio: {config['data']['positive_ratio']}")
        
        print(f"\n训练配置:")
        print(f"  Epochs: {config['training']['epochs']}")
        print(f"  Batch Size: {config['training']['batch_size']}")
        print(f"  动态 Loss: {config['training']['use_dynamic_loss']}")
        print(f"  Warmup Epochs: {config['training']['warmup_epochs']}")
        
        print(f"\n后处理配置:")
        print(f"  Multi-Threshold: {config['postprocessing']['multi_threshold']}")
        print(f"  Thresholds: {config['postprocessing']['thresholds']}")
        
        # 验证关键配置
        assert config['model']['in_channels'] == 5, "输入通道应该是 5"
        assert config['data']['dataset_type'] == 'ink_aware', "应该使用 ink_aware"
        assert config['data']['patch_size'] == [128, 128, 128], "Patch 应该是 128³"
        assert config['training']['use_dynamic_loss'] == True, "应该启用动态 Loss"
        
        print("\n✅ 测试通过")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("优化功能完整测试")
    print("="*60)
    
    results = []
    
    # 测试 1: Ink-only Sampling
    results.append(("Ink-only Sampling", test_ink_sampling()))
    
    # 测试 2: 多通道特征
    results.append(("多通道特征", test_multi_channel()))
    
    # 测试 3: 动态 Loss
    results.append(("动态 Loss 调度", test_dynamic_loss()))
    
    # 测试 4: 配置加载
    results.append(("优化配置文件", test_config_loading()))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n通过: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        print("\n下一步:")
        print("1. 准备真实数据")
        print("2. 在 AutoDL 上快速验证（8 epochs）")
        print("3. python train.py --config configs/autodl_dynunet_optimized.yaml")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")

    # 环境自检（MONAI / Torch / GPU）
    print("\n" + "-" * 60)
    print("环境自检 (MONAI / Torch / GPU)")
    print("-" * 60)
    try:
        import monai  # type: ignore
        import torch  # type: ignore

        print(f"MONAI 版本: {monai.__version__}")
        print(f"Torch 版本: {torch.__version__}")

        if not torch.cuda.is_available():
            print("❌ CUDA 不可用，请检查 GPU 驱动 / CUDA 安装")
        else:
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / 1e9
            print(f"GPU: {props.name}, 显存: {total_gb:.1f} GB")
            if props.total_memory <= 30e9:
                print("⚠️ GPU 显存 <= 30GB，可能无法安全跑 128³ Patch + 大模型")
            else:
                print("✅ GPU 显存满足 128³ Patch + DynUNet 训练需求")
    except Exception as e:  # pragma: no cover - 仅作运行环境提示
        print(f"⚠️ 环境自检出错: {e}")

    # 可选：检查后处理优化脚本是否可用
    print("\n" + "-" * 60)
    print("后处理优化脚本可用性检查 (optimize_postprocessing.py)")
    print("-" * 60)
    try:
        import optimize_postprocessing  # type: ignore

        print("✅ 成功导入 optimize_postprocessing 模块")
        if hasattr(optimize_postprocessing, "main"):
            print("   提示: 可在训练后运行 `python optimize_postprocessing.py` 对阈值/后处理做网格搜索")
        else:
            print("   注意: 模块中未找到 main() 函数，如需一键运行可后续添加入口函数")
    except Exception as e:  # pragma: no cover - 仅作运行环境提示
        print(f"⚠️ 无法导入 optimize_postprocessing: {e}")

    print("="*60)


if __name__ == "__main__":
    main()
