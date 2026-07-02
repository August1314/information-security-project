# 信息安全技术期末考查

2025 年《信息安全技术》课程期末考查项目。

**选题方向**：方向 2 — 复现两篇同领域 CCF-B 以上论文并对比优缺点

**研究领域**：数字水印技术 — 编码器-解码器图像水印

**选定论文**：
- MBRS (ACM MM 2021, CCF-B) — CNN 编码器-解码器 + 小批次 JPEG 压缩
- DWSF (ACM MM 2023, CCF-B) — 分散嵌入 + 分割同步 + 消息融合

**组员**：力航、苟一霏

## 快速开始

```bash
# 1. 创建虚拟环境
uv venv && source .venv/bin/activate

# 2. 安装依赖
uv pip install torch torchvision kornia numpy Pillow scipy tqdm opencv-python pyyaml crc8

# 3. 应用 MPS 适配（Mac Apple Silicon 必需）
python code/patch_mps.py

# 4. 验证安装
python -c "
import torch
print('MPS:', torch.backends.mps.is_available())
print('设备:', 'mps' if torch.backends.mps.is_available() else 'cpu')
"
```

## 项目结构

```
├── README.md
├── .gitignore
├── LICENSE
├── papers/              # 论文 PDF 及对比说明
├── code/
│   ├── patch_mps.py     # MPS 适配脚本
│   ├── MBRS/            # ACM MM 2021
│   └── DWSF/            # ACM MM 2023
├── report/              # 课程报告
└── results/             # 实验结果
```

## 实验流程

1. 准备 COCO2017 数据集
2. 训练 MBRS（`python code/MBRS/train.py`）
3. 训练 DWSF（`python code/DWSF/train_ed.py`）
4. 评估对比（PSNR、SSIM、BER，多种攻击鲁棒性）
5. 撰写报告

## 实验环境

- Mac M4 + MPS (Metal Performance Shaders)
- PyTorch 2.12.1, Python 3.13

## 截止日期

2025 年 7 月 13 日 23:59
