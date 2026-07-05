# 实验结果文档

## 文件结构

| 文件/目录 | 描述 |
|-----------|------|
|  | MBRS 训练好的模型权重（20.8M参数） |
|  | DWSF 训练好的模型权重（4.1M参数） |
|  | 所有实验数据的JSON格式 |
|  | 水印样本图片（18张） |
|  | 本文件 |

## 鲁棒性对比

| 攻击类型 | MBRS PSNR(dB) | MBRS BER | DWSF PSNR(dB) | DWSF BER |
|----------|:------------:|:--------:|:------------:|:--------:|
| 无攻击 | 33.8 | 0.0000 | 28.3 | 0.0000 |
| JPEG压缩 Q=90 | 34.5 | 0.0000 | 29.7 | 0.0202 |
| JPEG压缩 Q=50 | 30.0 | 0.4612 | 28.8 | 0.4929 |
| 高斯模糊 r=3 | 30.6 | 0.3195 | 29.2 | 0.4851 |
| 高斯模糊 r=5 | 28.4 | 0.5078 | 27.6 | 0.5226 |
| 高斯噪声 σ=0.02 | 32.9 | 0.0000 | 28.0 | 0.0006 |
| 高斯噪声 σ=0.05 | 29.9 | 0.0000 | 26.7 | 0.0006 |
| 高斯噪声 σ=0.10 | 25.6 | 0.0006 | 24.3 | 0.0024 |
| 裁剪 10% | 21.2 | 0.4746 | 21.3 | 0.4827 |
| 裁剪 25% | 17.3 | 0.4897 | 17.3 | 0.4833 |
| 亮度×2 | 16.6 | 0.0530 | 16.2 | 0.0214 |
| 亮度×0.5 | 16.5 | 0.0000 | 16.4 | 0.0732 |

- MBRS: **6/12** 种攻击下 BER<5%
- DWSF: **6/12** 种攻击下 BER<5%

## 分析要点

### 图像质量
- MBRS PSNR 更高（33.8 vs 28.3 dB），SSIM 也更高（0.84 vs 0.63），因为参数量是DWSF的5倍
- 两个模型Clean条件下BER均为0，水印提取完美
### 抗噪声
- 两个模型对高斯噪声都极其鲁棒（σ=0.10时BER<0.3%）
### 抗压缩
- JPEG轻压缩(Q=90)下MBRS表现更好（BER=0 vs 0.02）
- JPEG强压缩(Q=50)下两者都失效
### 抗模糊
- MBRS在弱模糊(r=3)下更好（BER=0.32 vs 0.49）
- 强模糊(r=5)下两者都失效
### 抗裁剪
- 裁剪是两者共同的弱点，即使10%裁剪也导致BER~0.48
### 抗亮度变化
- MBRS对亮度降低免疫(BER=0)，但对亮度翻倍敏感(BER=0.05)
- DWSF对亮度翻倍更好(BER=0.02)，但对亮度减半较弱(BER=0.07)
### 模型效率
- DWSF 仅用 4.1M参数（MBRS的1/5）达到了Clean条件下BER=0
- 在噪声和亮度变化下DWSF表现接近MBRS

## 图片清单

| 文件名 | 描述 |
|--------|------|
| dwsf_sample1_difference_x10.png | [DWSF] 样本1水印残差×10倍放大（展示编码引入的变化） |
| dwsf_sample1_original.png | [DWSF] 样本1原图（128×128测试图像） |
| dwsf_sample1_watermarked.png | [DWSF] 样本1嵌入水印后图像（肉眼与原始无差异） |
| dwsf_sample2_difference_x10.png | [DWSF] 样本2水印残差×10倍放大 |
| dwsf_sample2_original.png | [DWSF] 样本2原图 |
| dwsf_sample2_watermarked.png | [DWSF] 样本2嵌入水印后图像 |
| dwsf_sample3_difference_x10.png | [DWSF] 样本3水印残差×10倍放大 |
| dwsf_sample3_original.png | [DWSF] 样本3原图 |
| dwsf_sample3_watermarked.png | [DWSF] 样本3嵌入水印后图像 |
| mbrs_sample1_difference_x10.png | [MBRS] 样本1水印残差×10倍放大（展示编码引入的变化） |
| mbrs_sample1_original.png | [MBRS] 样本1原图（128×128测试图像） |
| mbrs_sample1_watermarked.png | [MBRS] 样本1嵌入水印后图像（肉眼与原始无差异） |
| mbrs_sample2_difference_x10.png | [MBRS] 样本2水印残差×10倍放大 |
| mbrs_sample2_original.png | [MBRS] 样本2原图 |
| mbrs_sample2_watermarked.png | [MBRS] 样本2嵌入水印后图像 |
| mbrs_sample3_difference_x10.png | [MBRS] 样本3水印残差×10倍放大 |
| mbrs_sample3_original.png | [MBRS] 样本3原图 |
| mbrs_sample3_watermarked.png | [MBRS] 样本3嵌入水印后图像 |

## 实验环境

- 设备: Apple M4 + MPS (Metal Performance Shaders)
- PyTorch 2.12.1, Python 3.13
- 训练数据: 300张合成结构图像（128×128）
- 测试数据: 50张合成结构图像
- MBRS消息长度: 64 bits, 参数量: 20.8M
- DWSF消息长度: 30 bits, 参数量: 4.1M
## 可视化图表清单

| 文件 | 类别 | 描述 |
|------|------|------|
| A1_psnr_curve.png | 训练过程 | PSNR收敛曲线 — 两个模型随epoch变化的PSNR对比 |
| A2_ber_curve.png | 训练过程 | BER收敛曲线 — 水印提取错误率下降过程 |
| A3_loss_curve.png | 训练过程 | Loss收敛曲线 — Encoder/Decoder训练损失分别展示 |
| B1_ber_bar.png | 鲁棒性对比 | BER攻击对比柱状图 — 12种攻击下两模型BER并排对比 |
| B2_psnr_bar.png | 鲁棒性对比 | PSNR攻击对比柱状图 — 12种攻击下图像质量变化 |
| B3_radar.png | 鲁棒性对比 | 鲁棒性雷达图 — 6种攻击(1-BER)评分雷达对比 |
| C1_watermark_samples.png | 图像可视化 | 水印样本全景图 — 3样本×2模型 原图/水印/残差对比 |
| C2_attack_effects.png | 图像可视化 | 攻击效果展示 — 水印图像在不同攻击下的视觉变化 |
| C3_architecture.png | 图像可视化 | 模型架构对比 — MBRS vs DWSF 架构框图 |
| D1_param_perf.png | 模型对比总结 | 参数vs性能散点图 — 参数量与PSNR/BER的权衡 |
| D2_summary_table.png | 模型对比总结 | 综合对比汇总表 — 所有维度指标的完整对比表格 |