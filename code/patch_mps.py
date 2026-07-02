#!/usr/bin/env python3
"""
MPS 适配脚本：修复 MBRS 和 DWSF 代码以支持 Apple Silicon Mac 运行。

用法:
    python code/patch_mps.py          # 应用补丁
    python code/patch_mps.py --undo   # 撤销补丁
"""

import os
import sys
import shutil
from pathlib import Path

BASE = Path(__file__).parent

BACKUP_DIR = BASE / ".mps_backup"


def backup(filepath):
    """创建备份"""
    dest = BACKUP_DIR / filepath.relative_to(BASE)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(filepath, dest)
    print(f"  backup: {filepath.relative_to(BASE)}")


def patch_file(filepath, replacements):
    """对文件执行字符串替换"""
    content = filepath.read_text()
    original = content
    for old, new in replacements:
        content = content.replace(old, new)
    if content != original:
        filepath.write_text(content)
        return True
    return False


def apply_patches():
    print("=" * 50)
    print("应用 MPS 适配补丁")
    print("=" * 50)

    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    BACKUP_DIR.mkdir()

    # ---- MBRS ----
    print("\n[MBRS]")

    # 1. test.py - 设备检测 + pin_memory
    f = BASE / "MBRS/test.py"
    backup(f)
    patch_file(f, [
        ('device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
         'device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))'),
        (', pin_memory=True', ''),
    ])
    print("  test.py: 设备检测 + pin_memory 已修复")

    # 2. train.py - 设备检测 + pin_memory
    f = BASE / "MBRS/train.py"
    backup(f)
    patch_file(f, [
        ('device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
         'device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))'),
        (', pin_memory=True', ''),
    ])
    print("  train.py: 设备检测 + pin_memory 已修复")

    # 3. Network.py - 移除 DataParallel
    f = BASE / "MBRS/network/Network.py"
    backup(f)
    patch_file(f, [
        ('\t\tself.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)\n\t\tself.discriminator = torch.nn.DataParallel(self.discriminator)',
         '\t\t# MPS: DataParallel 已移除，单 GPU 不需要\n\t\t# self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)\n\t\t# self.discriminator = torch.nn.DataParallel(self.discriminator)'),
    ])
    print("  Network.py: DataParallel 已移除")

    # ---- DWSF ----
    print("\n[DWSF]")

    mps_device_line = 'device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))'

    # 1. train_ed.py - 设备检测
    f = BASE / "DWSF/train_ed.py"
    backup(f)
    patch_file(f, [
        ('device = torch.device(\'cuda\' if torch.cuda.is_available() else "cpu")', mps_device_line),
    ])
    print("  train_ed.py: 设备检测已修复")

    # 2. evaluate.py
    f = BASE / "DWSF/evaluate.py"
    backup(f)
    patch_file(f, [
        ('device = torch.device(\'cuda\' if torch.cuda.is_available() else "cpu")', mps_device_line),
    ])
    print("  evaluate.py: 设备检测已修复")

    # 3. generate_segdata.py
    f = BASE / "DWSF/generate_segdata.py"
    backup(f)
    patch_file(f, [
        ("device = torch.device('cuda:0' if torch.cuda.is_available() else \"cpu\")", mps_device_line),
    ])
    print("  generate_segdata.py: 设备检测已修复")

    # 4. train_seg.py
    f = BASE / "DWSF/train_seg.py"
    backup(f)
    patch_file(f, [
        (', target.cuda()', ', target.to(device)'),
        ('image.cuda()', 'image.to(device)'),
        (', data.cuda(), target.cuda()', ', data.to(device), target.to(device)'),
        ("U2NETP(mode='train').cuda()", "U2NETP(mode='train').to(device)"),
    ])
    print("  train_seg.py: .cuda() -> .to(device) 已修复")

    # 5. utils/seg.py
    f = BASE / "DWSF/utils/seg.py"
    backup(f)
    patch_file(f, [
        ('device = torch.device(\'cuda\' if torch.cuda.is_available() else "cpu")', mps_device_line),
        ('image.cuda()', 'image.to(device)'),
        ('image_tensor.cuda()', 'image_tensor.to(device)'),
    ])
    print("  utils/seg.py: 设备检测 + .cuda() 已修复")

    # 6. utils/img.py
    f = BASE / "DWSF/utils/img.py"
    backup(f)
    patch_file(f, [
        ('device = torch.device("cuda" if torch.cuda.is_available() else "cpu")', mps_device_line),
    ])
    print("  utils/img.py: 设备检测已修复")

    # 7. utils/util.py - torch.cuda.manual_seed 兜底
    f = BASE / "DWSF/utils/util.py"
    backup(f)
    content = f.read_text()
    if 'torch.cuda.manual_seed_all(seed)' in content and 'try:' not in content.split('torch.cuda.manual_seed_all')[0][-50:]:
        content = content.replace(
            'torch.cuda.manual_seed_all(seed)\n    torch.cuda.manual_seed(seed)',
            'try:\n        torch.cuda.manual_seed_all(seed)\n        torch.cuda.manual_seed(seed)\n    except Exception:\n        pass  # MPS: CUDA manual_seed 不可用，跳过'
        )
        f.write_text(content)
        print("  utils/util.py: CUDA manual_seed 已兜底")

    # 8. train_ed.py - CUDA_LAUNCH_BLOCKING 移除
    f = BASE / "DWSF/train_ed.py"
    backup(f)
    patch_file(f, [
        ("os.environ['CUDA_LAUNCH_BLOCKING'] = '1'", "# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # MPS: 不需要"),
    ])

    print("\n" + "=" * 50)
    print("MPS 适配完成！备份保存在 code/.mps_backup/")
    print("如需还原: python code/patch_mps.py --undo")
    print("=" * 50)


def undo_patches():
    print("=" * 50)
    print("撤销 MPS 补丁，恢复原始文件")
    print("=" * 50)

    if not BACKUP_DIR.exists():
        print("未找到备份目录，无需撤销。")
        return

    for backup_file in BACKUP_DIR.rglob("*"):
        if backup_file.is_file():
            target = BASE / backup_file.relative_to(BACKUP_DIR)
            shutil.copy2(backup_file, target)
            print(f"  恢复: {target.relative_to(BASE)}")

    shutil.rmtree(BACKUP_DIR)
    print("\n已恢复所有原始文件。")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--undo":
        undo_patches()
    else:
        apply_patches()
