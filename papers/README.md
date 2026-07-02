# 候选论文

## 选定方案

方向 2（复现两篇 + 对比），数字水印 — 编码器-解码器图像水印子领域。

## 论文清单

| # | 论文 | 会议 | 年份 | CCF等级 | 代码 |
|---|------|------|------|---------|------|
| 1 | MBRS | ACM MM | 2021 | B | [GitHub](https://github.com/jzyustc/MBRS) |
| 2 | DWSF | ACM MM | 2023 | B | [GitHub](https://github.com/bytedance/DWSF) |

## 对比维度

两个方法都在鲁棒图像水印领域，都基于编码器-解码器架构，但技术路线不同：

- **MBRS**：HiDDeN 风格 CNN 编码器-解码器，用真实+模拟 JPEG 小批次训练增强抗压缩鲁棒性
- **DWSF**：分散嵌入 + 分割同步 + 消息融合，支持任意分辨率图像，对几何攻击更鲁棒

### 评估指标

- PSNR / SSIM（图像质量）
- BER / Bit Accuracy（消息提取准确率）
- 鲁棒性：JPEG 压缩、裁剪、旋转、高斯模糊、噪声、色彩变换

### 实验环境

- Python 3.8+ / PyTorch 2.x
- 数据集：COCO2017
- 推理训练均可：Mac M4 MPS
