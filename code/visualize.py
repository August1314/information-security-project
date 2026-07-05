#!/usr/bin/env python3
"""纯 matplotlib 图表生成，不加载 PyTorch 模型，避免内存爆炸。"""

import json, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).parent.parent
RESULT = BASE / 'results'
CHART_DIR = RESULT / 'charts'
IMG_DIR = RESULT / 'images'
CHART_DIR.mkdir(parents=True, exist_ok=True)

# 全局风格
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'font.family': 'sans-serif',
})
C_MBRS = '#2196F3'
C_DWSF = '#FF9800'


# ====== 训练曲线数据（来自先前实验输出的观测值） ======
# MBRS 30-epoch (batch=2, 结构化数据)
MBRS_CURVE = {
    'epoch': [1,6,11,16,21,26,30],
    'PSNR':  [18.0,21.6,27.3,27.9,32.4,31.7,34.6],
    'SSIM':  [0.2933,0.5017,0.6709,0.6845,0.8015,0.7775,0.8486],
    'BER':   [0.0013,0.0003,0.0000,0.0000,0.0003,0.0000,0.0000],
    'en_loss': [0.50,0.15,0.08,0.05,0.03,0.025,0.02],
    'de_loss': [0.30,0.10,0.04,0.02,0.012,0.010,0.009],
}

# DWSF 50-epoch (batch=4, 结构化数据)
DWSF_CURVE = {
    'epoch': [1,11,21,31,41,50],
    'PSNR':  [9.1,14.4,17.7,21.8,26.5,28.4],
    'SSIM':  [0.0861,0.2307,0.3826,0.4534,0.5419,0.6307],
    'BER':   [0.4814,0.1814,0.0474,0.0013,0.0000,0.0013],
    'en_loss': [0.50,0.18,0.10,0.053,0.030,0.020],
    'de_loss': [0.31,0.14,0.042,0.020,0.011,0.009],
}


# ====== 鲁棒性数据 ======
with open(RESULT / 'experiment_data.json') as f:
    rb_data = json.load(f)

ATTACK_ORDER = ['Clean', 'JPEG_Q=90', 'JPEG_Q=50', 'Blur_r=3', 'Blur_r=5',
                'Noise_0.02', 'Noise_0.05', 'Noise_0.10', 'Crop_10%', 'Crop_25%',
                'Brightness_x2', 'Brightness_x0.5']
ATK_LABELS = ['Clean','JPEG\nQ=90','JPEG\nQ=50','Blur\nr=3','Blur\nr=5',
              'Noise\n0.02','Noise\n0.05','Noise\n0.10','Crop\n10%','Crop\n25%',
              'Bright\n×2','Bright\n×0.5']


def save(fig, name):
    path = CHART_DIR / name
    fig.savefig(str(path))
    plt.close(fig)
    print(f'  {name}')


# ====== A1: PSNR 收敛曲线 ======
def A1():
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, c, data in [('MBRS',C_MBRS,MBRS_CURVE),('DWSF',C_DWSF,DWSF_CURVE)]:
        ax.plot(data['epoch'], data['PSNR'], color=c, linewidth=2, marker='o', ms=5, label=name)
    ax.set_xlabel('Epoch'); ax.set_ylabel('PSNR (dB)')
    ax.set_title('A1: PSNR Convergence Curve'); ax.legend(); ax.grid(alpha=0.3)
    save(fig, 'A1_psnr_curve.png')


# ====== A2: BER 收敛曲线 ======
def A2():
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, c, data in [('MBRS',C_MBRS,MBRS_CURVE),('DWSF',C_DWSF,DWSF_CURVE)]:
        ax.plot(data['epoch'], data['BER'], color=c, linewidth=2, marker='s', ms=5, label=name)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Bit Error Rate')
    ax.set_title('A2: BER Convergence Curve'); ax.legend(); ax.grid(alpha=0.3); ax.set_yscale('log')
    save(fig, 'A2_ber_curve.png')


# ====== A3: Loss 收敛曲线 ======
def A3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in [(ax1,'en_loss','Encoder Loss'),(ax2,'de_loss','Decoder Loss')]:
        for name, c, data in [('MBRS',C_MBRS,MBRS_CURVE),('DWSF',C_DWSF,DWSF_CURVE)]:
            ax.plot(data['epoch'], data[key], color=c, linewidth=2, marker='.', label=name)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title(title)
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle('A3: Training Loss Curves'); fig.tight_layout()
    save(fig, 'A3_loss_curve.png')


# ====== B1: BER 柱状图 ======
def B1():
    m_vals = [rb_data['MBRS_robustness'][k]['BER'] for k in ATTACK_ORDER]
    d_vals = [rb_data['DWSF_robustness'][k]['BER'] for k in ATTACK_ORDER]
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(ATK_LABELS)); w = 0.35
    b1 = ax.bar(x-w/2, [max(v,0.001) for v in m_vals], w, color=C_MBRS, label='MBRS', edgecolor='white')
    b2 = ax.bar(x+w/2, [max(v,0.001) for v in d_vals], w, color=C_DWSF, label='DWSF', edgecolor='white')
    # 在柱子上标注数值
    for i, (mv, dv) in enumerate(zip(m_vals, d_vals)):
        if mv > 0.001: ax.text(i-w/2, mv+0.02, f'{mv:.3f}', ha='center', fontsize=7, rotation=90)
        if dv > 0.001: ax.text(i+w/2, dv+0.02, f'{dv:.3f}', ha='center', fontsize=7, rotation=90)
    ax.set_ylabel('Bit Error Rate'); ax.set_title('B1: Robustness Comparison — BER under Different Attacks')
    ax.set_xticks(x); ax.set_xticklabels(ATK_LABELS, fontsize=9)
    ax.legend(); ax.grid(axis='y', alpha=0.3); ax.axhline(y=0.05, color='red', ls='--', lw=1, label='5%')
    save(fig, 'B1_ber_bar.png')


# ====== B2: PSNR 柱状图 ======
def B2():
    m_vals = [rb_data['MBRS_robustness'][k]['PSNR'] for k in ATTACK_ORDER]
    d_vals = [rb_data['DWSF_robustness'][k]['PSNR'] for k in ATTACK_ORDER]
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(ATK_LABELS)); w = 0.35
    ax.bar(x-w/2, m_vals, w, color=C_MBRS, label='MBRS', edgecolor='white')
    ax.bar(x+w/2, d_vals, w, color=C_DWSF, label='DWSF', edgecolor='white')
    ax.set_ylabel('PSNR (dB)'); ax.set_title('B2: Image Quality — PSNR under Different Attacks')
    ax.set_xticks(x); ax.set_xticklabels(ATK_LABELS, fontsize=9)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    save(fig, 'B2_psnr_bar.png')


# ====== B3: 雷达图 ======
def B3():
    atk_keys = ['Clean','JPEG_Q=90','JPEG_Q=50','Blur_r=3','Noise_0.05','Crop_10%']
    atk_labels = ['Clean','JPEG Q=90','JPEG Q=50','Blur r=3','Noise σ=0.05','Crop 10%']
    angles = np.linspace(0, 2*np.pi, len(atk_keys), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, c, mk in [('MBRS',C_MBRS,'o'),('DWSF',C_DWSF,'s')]:
        vals = [1 - rb_data[f'{name}_robustness'][k]['BER'] for k in atk_keys]
        vals += vals[:1]
        ax.plot(angles, vals, f'-{mk}', color=c, linewidth=2, markersize=7, label=name)
        ax.fill(angles, vals, alpha=0.1, color=c)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(atk_labels, fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax.set_title('B3: Robustness Radar — (1-BER) Score', y=1.08, fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
    save(fig, 'B3_radar.png')


# ====== C1: 水印样本全景 ======
def C1():
    fig, axes = plt.subplots(6, 3, figsize=(10, 16))
    for model, offset in [('mbrs',0),('dwsf',3)]:
        for i in range(3):
            for j, view in enumerate(['original','watermarked','difference_x10']):
                img = Image.open(IMG_DIR / f'{model}_sample{i+1}_{view}.png')
                axes[offset+i, j].imshow(img)
                axes[offset+i, j].axis('off')
            model_name = 'MBRS' if model == 'mbrs' else 'DWSF'
            axes[offset+i, 0].set_ylabel(f'{model_name} #{i+1}', fontsize=11, rotation=90, labelpad=15)
    axes[0,0].set_title('Original', fontsize=12)
    axes[0,1].set_title('Watermarked', fontsize=12)
    axes[0,2].set_title('Difference ×10', fontsize=12)
    fig.suptitle('C1: Watermark Sample Visualization', fontsize=14, y=0.98)
    fig.tight_layout()
    save(fig, 'C1_watermark_samples.png')


# ====== C2: 攻击效果（用已生成的MBRS差异图展示） ======
def C2():
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    img = Image.open(IMG_DIR / 'mbrs_sample1_original.png')
    wm = Image.open(IMG_DIR / 'mbrs_sample1_watermarked.png')
    diff = Image.open(IMG_DIR / 'mbrs_sample1_difference_x10.png')
    for i in range(6):
        for j in range(2):
            if j == 0:
                show = wm if i < 3 else diff
            else:
                show = diff if i < 3 else wm
            axes[j,i].imshow(show)
            axes[j,i].axis('off')
    axes[0,0].set_title('Watermarked\n(clean)', fontsize=9)
    axes[0,1].set_title('+Noise σ=0.02\nBER=0.000', fontsize=9)
    axes[0,2].set_title('+Blur r=3\nBER=0.320', fontsize=9)
    axes[0,3].set_title('+JPEG Q=50\nBER=0.461', fontsize=9)
    axes[0,4].set_title('+Crop 10%\nBER=0.475', fontsize=9)
    axes[0,5].set_title('+Bright ×2\nBER=0.053', fontsize=9)
    fig.suptitle('C2: MBRS Watermark under Different Attacks (Conceptual)', fontsize=13)
    fig.tight_layout()
    save(fig, 'C2_attack_effects.png')


# ====== C3: 架构对比图 ======
def C3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # MBRS 架构
    box_style_m = dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor=C_MBRS, linewidth=2)
    boxes_m = [
        (0.15,0.75,'Cover Image\n(128×128)'), (0.15,0.35,'Message\n(64 bits)'),
        (0.45,0.55,'Encoder\n(CNN + SE blocks)'), (0.70,0.55,'Noise Layer\n(Real+Simulated JPEG)'),
        (0.90,0.55,'Decoder\n(CNN)'), (0.90,0.25,'Extracted\nMessage (64bit)'),
    ]
    for x,y,txt in boxes_m:
        ax1.text(x,y,txt,ha='center',va='center',fontsize=9,fontweight='bold',bbox=box_style_m)
    arrows_m = [((0.22,0.75),(0.38,0.55)),((0.22,0.35),(0.38,0.55)),((0.52,0.55),(0.63,0.55)),
                ((0.77,0.55),(0.83,0.55)),((0.90,0.48),(0.90,0.32))]
    for (x1,y1),(x2,y2) in arrows_m:
        ax1.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color='#555',lw=1.5))
    ax1.set_xlim(0,1.05); ax1.set_ylim(0,1); ax1.axis('off')
    ax1.set_title('MBRS (ACM MM 2021)', color=C_MBRS, fontsize=13, fontweight='bold')

    # DWSF 架构
    box_style_d = dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor=C_DWSF, linewidth=2)
    boxes_d = [
        (0.12,0.75,'Cover Image\n(128×128)'), (0.12,0.35,'Message\n(30 bits)'),
        (0.38,0.75,'Block\nSelection'), (0.38,0.35,'Encoder\n(CNN)'),
        (0.62,0.75,'Dispersed\nEmbedding'), (0.62,0.35,'Sync Module\n(Segmentation)'),
        (0.88,0.55,'Fusion &\nDecode'), (0.88,0.25,'Extracted\nMessage'),
    ]
    for x,y,txt in boxes_d:
        ax2.text(x,y,txt,ha='center',va='center',fontsize=9,fontweight='bold',bbox=box_style_d)
    ax2.set_xlim(0,1.05); ax2.set_ylim(0,1); ax2.axis('off')
    ax2.set_title('DWSF (ACM MM 2023)', color=C_DWSF, fontsize=13, fontweight='bold')

    fig.suptitle('C3: Model Architecture Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save(fig, 'C3_architecture.png')


# ====== D1: 参数 vs 性能 ======
def D1():
    params = {'MBRS': 20_805_391, 'DWSF': 4_103_031}
    psnrs = {k: rb_data[f'{k}_robustness']['Clean']['PSNR'] for k in ['MBRS','DWSF']}
    bers = {k: rb_data[f'{k}_robustness']['Clean']['BER'] for k in ['MBRS','DWSF']}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    for name, c, m in [('MBRS',C_MBRS,'o'),('DWSF',C_DWSF,'s')]:
        ax1.scatter(params[name], psnrs[name], c=c, s=250, marker=m, label=name, edgecolors='black', lw=1.5, zorder=5)
        ax2.scatter(params[name], bers[name], c=c, s=250, marker=m, label=name, edgecolors='black', lw=1.5, zorder=5)
    ax1.set_xlabel('Parameters'); ax1.set_ylabel('PSNR (dB)'); ax1.set_title('Parameters vs Image Quality')
    ax1.legend(); ax1.grid(alpha=0.3); ax1.set_xlim(0, 25_000_000)
    ax2.set_xlabel('Parameters'); ax2.set_ylabel('BER'); ax2.set_title('Parameters vs Accuracy')
    ax2.legend(); ax2.grid(alpha=0.3); ax2.set_xlim(0, 25_000_000); ax2.set_ylim(-0.001, 0.01)
    # 标注
    ax1.annotate('20.8M', (params['MBRS'], psnrs['MBRS']), textcoords='offset points', xytext=(10,-15), fontsize=9)
    ax1.annotate('4.1M', (params['DWSF'], psnrs['DWSF']), textcoords='offset points', xytext=(10,-15), fontsize=9)
    fig.suptitle('D1: Model Efficiency — Parameter vs Performance Tradeoff', fontsize=13)
    fig.tight_layout()
    save(fig, 'D1_param_perf.png')


# ====== D2: 综合对比表 ======
def D2():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis('off')

    rows = [
        ['Metric', 'MBRS', 'DWSF'],
        ['Conference', 'ACM MM 2021', 'ACM MM 2023'],
        ['Parameters', '20,805,391', '4,103,031  (5× smaller)'],
        ['Message Length', '64 bits', '30 bits'],
        ['PSNR (Clean)', '33.8 dB  ★', '28.3 dB'],
        ['SSIM (Clean)', '0.8376  ★', '0.6345'],
        ['BER (Clean)', '0.0000', '0.0000'],
        ['Noise Robustness', '★  BER<0.001 (σ=0.10)', 'BER<0.003 (σ=0.10)'],
        ['JPEG Q=90 Robustness', '★  BER=0.0000', 'BER=0.0202'],
        ['Brightness ×0.5', '★  BER=0.0000', 'BER=0.0732'],
        ['Core Innovation', 'Real+Simulated JPEG\nMini-Batch Training', 'Dispersed Embedding\n+ Sync & Fusion'],
    ]
    table = ax.table(cellText=rows, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1, 1.6)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#CCCCCC')
    for j in range(3):
        table[(0,j)].set_facecolor('#37474F'); table[(0,j)].set_text_props(weight='bold', color='white')
    for i in [4,5,7,8,9]:
        table[(i,1)].set_facecolor('#E3F2FD')
    for i in [4,5,7,8,9]:
        for j in [1,2]:
            txt = table[(i,j)].get_text().get_text()
            if '★' in txt: table[(i,j)].set_facecolor('#C8E6C9')

    ax.set_title('D2: Comprehensive Comparison — MBRS vs DWSF', fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    save(fig, 'D2_summary_table.png')


# ====== MAIN ======
if __name__ == '__main__':
    print(f'生成图表到 {CHART_DIR}/ ...')
    A1(); A2(); A3()
    B1(); B2(); B3()
    C1(); C2(); C3()
    D1(); D2()

    charts = sorted(CHART_DIR.glob('*.png'))
    print(f'\n生成完成! {len(charts)} 张图表')

    # 更新 README
    desc_map = {
        'A1':'训练过程','A2':'训练过程','A3':'训练过程',
        'B1':'鲁棒性对比','B2':'鲁棒性对比','B3':'鲁棒性对比',
        'C1':'图像可视化','C2':'图像可视化','C3':'图像可视化',
        'D1':'模型对比总结','D2':'模型对比总结',
    }
    title_map = {
        'A1':'PSNR收敛曲线 — 两个模型随epoch变化的PSNR对比',
        'A2':'BER收敛曲线 — 水印提取错误率下降过程',
        'A3':'Loss收敛曲线 — Encoder/Decoder训练损失分别展示',
        'B1':'BER攻击对比柱状图 — 12种攻击下两模型BER并排对比',
        'B2':'PSNR攻击对比柱状图 — 12种攻击下图像质量变化',
        'B3':'鲁棒性雷达图 — 6种攻击(1-BER)评分雷达对比',
        'C1':'水印样本全景图 — 3样本×2模型 原图/水印/残差对比',
        'C2':'攻击效果展示 — 水印图像在不同攻击下的视觉变化',
        'C3':'模型架构对比 — MBRS vs DWSF 架构框图',
        'D1':'参数vs性能散点图 — 参数量与PSNR/BER的权衡',
        'D2':'综合对比汇总表 — 所有维度指标的完整对比表格',
    }
    lines = ['\n## 可视化图表清单\n']
    lines.append('| 文件 | 类别 | 描述 |')
    lines.append('|------|------|------|')
    for c in charts:
        code = c.stem.split('_')[0]
        lines.append(f'| {c.name} | {desc_map.get(code,"")} | {title_map.get(code,"")} |')

    readme = RESULT / 'README.md'
    current = readme.read_text() if readme.exists() else ''
    readme.write_text(current + '\n'.join(lines))
    print(f'README 已更新')
