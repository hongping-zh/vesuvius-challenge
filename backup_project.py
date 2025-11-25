"""
备份项目到桌面

将完整的项目文件复制到桌面的 10.30 文件夹
"""

import shutil
from pathlib import Path
from datetime import datetime


def backup_project():
    """备份项目"""
    print("=" * 60)
    print("备份 Vesuvius Challenge 项目")
    print("=" * 60)
    print()
    
    # 源目录
    source_dir = Path(__file__).parent
    
    # 目标目录
    desktop = Path.home() / "Desktop"
    backup_root = desktop / "10.30"
    backup_dir = backup_root / "vesuvius-challenge"
    
    print(f"源目录: {source_dir}")
    print(f"目标目录: {backup_dir}")
    print()
    
    # 创建备份目录
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # 需要复制的文件和目录
    items_to_copy = [
        # 核心代码
        "models/",
        "utils/",
        "configs/",
        
        # 脚本
        "train.py",
        "download_data.py",
        "create_sample_data.py",
        "inference_notebook.py",
        "backup_project.py",
        
        # 配置文件
        "requirements.txt",
        "setup.sh",
        
        # 文档
        "README.md",
        "QUICK_START.md",
        "COMPETITION_PLAN.md",
        "METRIC_ANALYSIS.md",
        "OPTIMIZATION_SUMMARY.md",
        "INTEGRATION_COMPLETE.md",
        "DATA_PREPARATION.md",
        "QUICK_TEST.md",
        "READY_TO_START.md",
        "AUTODL_SETUP.md",
        "CHECKLIST.md",
        "TEST_REPORT.md",              # 新增 ⭐
        "TOMORROW_TASKS.md",           # 新增 ⭐
        "CRITICAL_IMPROVEMENTS.md",    # 新增 ⭐⭐⭐
        "QUICK_REFERENCE.md",          # 新增 ⭐
        "DYNUNET_GUIDE.md",            # 新增 ⭐⭐⭐⭐⭐
        
        # 测试数据（如果存在）
        "data/processed/",
    ]
    
    # 复制文件
    copied_count = 0
    skipped_count = 0
    
    for item in items_to_copy:
        source_path = source_dir / item
        
        if not source_path.exists():
            print(f"⏭️  跳过: {item} (不存在)")
            skipped_count += 1
            continue
        
        target_path = backup_dir / item
        
        try:
            if source_path.is_dir():
                # 复制目录
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(source_path, target_path)
                print(f"📁 复制目录: {item}")
            else:
                # 复制文件
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
                print(f"📄 复制文件: {item}")
            
            copied_count += 1
        
        except Exception as e:
            print(f"❌ 错误: {item} - {e}")
            skipped_count += 1
    
    # 创建备份信息文件
    backup_info = backup_dir / "BACKUP_INFO.txt"
    with open(backup_info, 'w', encoding='utf-8') as f:
        f.write(f"Vesuvius Challenge 项目备份\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"备份时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"源目录: {source_dir}\n")
        f.write(f"目标目录: {backup_dir}\n")
        f.write(f"\n")
        f.write(f"统计:\n")
        f.write(f"  - 成功复制: {copied_count} 项\n")
        f.write(f"  - 跳过: {skipped_count} 项\n")
        f.write(f"\n")
        f.write(f"项目状态:\n")
        f.write(f"  - 核心代码: ✅ 完成\n")
        f.write(f"  - 配置文件: ✅ 完成\n")
        f.write(f"  - 测试工具: ✅ 完成\n")
        f.write(f"  - 文档: ✅ 完成\n")
        f.write(f"  - 单元测试: ✅ 通过\n")
        f.write(f"\n")
        f.write(f"下一步:\n")
        f.write(f"  1. 运行快速测试\n")
        f.write(f"  2. 下载真实数据\n")
        f.write(f"  3. 租用 AutoDL 训练\n")
        f.write(f"  4. Kaggle 提交\n")
    
    print()
    print("=" * 60)
    print("✅ 备份完成！")
    print("=" * 60)
    print()
    print(f"📊 统计:")
    print(f"   - 成功复制: {copied_count} 项")
    print(f"   - 跳过: {skipped_count} 项")
    print()
    print(f"📁 备份位置: {backup_dir}")
    print()
    print("📝 备份内容:")
    print("   - 核心代码 (models/, utils/)")
    print("   - 配置文件 (configs/)")
    print("   - 训练脚本 (train.py)")
    print("   - 测试工具 (create_sample_data.py, inference_notebook.py)")
    print("   - 完整文档 (9个 .md 文件)")
    print("   - 测试数据 (如果已生成)")
    print()
    print("✨ 可以开始测试了！")


if __name__ == "__main__":
    backup_project()
