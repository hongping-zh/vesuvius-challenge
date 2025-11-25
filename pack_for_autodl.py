"""
打包项目用于上传到 AutoDL
排除大文件和临时文件
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def pack_project():
    """打包项目"""
    print("="*60)
    print("打包项目用于 AutoDL")
    print("="*60)
    
    # 项目根目录
    project_dir = Path(__file__).parent
    
    # 输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = project_dir.parent / f"vesuvius-challenge-{timestamp}.tar.gz"
    
    # 需要包含的文件和目录
    include_patterns = [
        "models/*.py",
        "utils/*.py",
        "configs/*.yaml",
        "*.py",
        "*.sh",
        "*.md",
        "requirements.txt",
    ]
    
    # 需要排除的目录
    exclude_dirs = [
        "data",
        "models/checkpoints*",
        "logs",
        "__pycache__",
        ".git",
        ".vscode",
        "*.pth",
        "*.tar.gz",
    ]
    
    print(f"\n📦 创建压缩包: {output_file.name}")
    print(f"   位置: {output_file.parent}")
    
    # 创建临时目录
    temp_dir = project_dir.parent / "temp_pack"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    target_dir = temp_dir / "vesuvius-challenge"
    target_dir.mkdir()
    
    # 复制文件
    copied_count = 0
    
    print(f"\n📋 复制文件...")
    
    # 复制 Python 文件
    for py_file in project_dir.glob("*.py"):
        shutil.copy2(py_file, target_dir / py_file.name)
        print(f"   ✓ {py_file.name}")
        copied_count += 1
    
    # 复制 Shell 脚本
    for sh_file in project_dir.glob("*.sh"):
        shutil.copy2(sh_file, target_dir / sh_file.name)
        print(f"   ✓ {sh_file.name}")
        copied_count += 1
    
    # 复制 Markdown 文件
    for md_file in project_dir.glob("*.md"):
        shutil.copy2(md_file, target_dir / md_file.name)
        print(f"   ✓ {md_file.name}")
        copied_count += 1
    
    # 复制 requirements.txt
    req_file = project_dir / "requirements.txt"
    if req_file.exists():
        shutil.copy2(req_file, target_dir / "requirements.txt")
        print(f"   ✓ requirements.txt")
        copied_count += 1
    
    # 复制 models 目录
    models_src = project_dir / "models"
    models_dst = target_dir / "models"
    models_dst.mkdir()
    for py_file in models_src.glob("*.py"):
        shutil.copy2(py_file, models_dst / py_file.name)
        print(f"   ✓ models/{py_file.name}")
        copied_count += 1
    
    # 复制 utils 目录
    utils_src = project_dir / "utils"
    utils_dst = target_dir / "utils"
    utils_dst.mkdir()
    for py_file in utils_src.glob("*.py"):
        shutil.copy2(py_file, utils_dst / py_file.name)
        print(f"   ✓ utils/{py_file.name}")
        copied_count += 1
    
    # 复制 configs 目录
    configs_src = project_dir / "configs"
    configs_dst = target_dir / "configs"
    configs_dst.mkdir()
    for yaml_file in configs_src.glob("*.yaml"):
        shutil.copy2(yaml_file, configs_dst / yaml_file.name)
        print(f"   ✓ configs/{yaml_file.name}")
        copied_count += 1
    
    # 创建压缩包
    print(f"\n🗜️  压缩中...")
    shutil.make_archive(
        str(output_file.with_suffix('')),
        'gztar',
        temp_dir
    )
    
    # 清理临时目录
    shutil.rmtree(temp_dir)
    
    # 获取文件大小
    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
    
    print(f"\n✅ 打包完成！")
    print(f"   文件: {output_file.name}")
    print(f"   大小: {file_size:.2f} MB")
    print(f"   文件数: {copied_count}")
    
    print(f"\n📤 上传到 AutoDL:")
    print(f"   1. 在 AutoDL 上创建实例")
    print(f"   2. 使用文件上传功能上传: {output_file.name}")
    print(f"   3. 解压: tar -xzf {output_file.name}")
    print(f"   4. 进入目录: cd vesuvius-challenge")
    print(f"   5. 运行设置脚本: bash autodl_setup.sh")
    
    print("="*60)


if __name__ == "__main__":
    pack_project()
