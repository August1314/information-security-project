# 候选论文

## 选定方案

方向 2（复现两篇 + 对比），数字水印 — 扩散模型图像水印子领域。

## 论文清单

| # | 论文 | 会议 | 年份 | CCF等级 | 代码 |
|---|------|------|------|---------|------|
| 1 | Tree-Ring Watermarks | NeurIPS | 2023 | A | [GitHub](https://github.com/YuxinWenRick/tree-ring-watermark) |
| 2 | Gaussian Shading | CVPR | 2024 | A | [GitHub](https://github.com/bsmhmmlf/Gaussian-Shading) |

## 对比维度

两个方法都在扩散模型图像水印领域，都采用即插即用（无需训练）方案，共享相同的评估框架：

- **Tree-Ring**：在初始噪声傅里叶域嵌入环形图案，用 L1 距离检测
- **Gaussian Shading**：用高斯分布编码 + ChaCha20 加密嵌入水印，可证性能无损

### 评估指标

- CLIP Score（图像-文本相似度）
- FID（生成质量）
- AUC / TPR@1%FPR（检测准确率）
- 鲁棒性：JPEG 压缩、旋转、裁剪、高斯模糊、亮度调整

### 实验环境

- Stable Diffusion 2.1
- COCO 5000 张评估集
- PyTorch 1.13 + CUDA GPU
