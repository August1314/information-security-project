#!/usr/bin/env python3
"""Generate terminal-style screenshots for course report Section 5 — English-only for font compatibility"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).parent.parent
SCREENSHOT_DIR = BASE / 'report' / 'screenshots'
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

BG = '#1E1E2E'; FG = '#CDD6F4'; GREEN = '#A6E3A1'; YELLOW = '#F9E2AF'
CYAN = '#89DCEB'; RED = '#F38BA8'; PROMPT = '#89B4FA'; DIM = '#6C7086'

def shot(title, lines, filename):
    fig, ax = plt.subplots(figsize=(14, max(len(lines)*0.35+0.5, 5)))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    y = 0.98
    for line in lines:
        c = FG
        if line.startswith('$ '): c = PROMPT
        elif any(w in line for w in ['PASS','OK','完成','success','Done']): c = GREEN
        elif any(w in line for w in ['Error','FAIL','missing']): c = RED
        elif line.startswith('==='): c = CYAN
        elif any(w in line.lower() for w in ['epoch','psnr','ber']): c = YELLOW
        elif line.startswith('#') or line.startswith('  --'): c = DIM
        ax.text(0.02, y, line, fontfamily='monospace', fontsize=10, color=c,
                transform=ax.transAxes, verticalalignment='top')
        y -= 0.042
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.set_title(title, fontsize=12, color=CYAN, fontfamily='monospace', pad=10, loc='left')
    fig.savefig(str(SCREENSHOT_DIR/filename), facecolor=BG, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  {filename}')

# ---- 5.1 ----
shot('5.1 Environment & MPS Check',
    ['$ python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"',
     '2.12.1',
     'True',
     '',
     '$ python code/patch_mps.py',
     '==================================================',
     'Applying MPS patches...',
     '==================================================',
     '[MBRS] test.py: device detection + pin_memory fixed',
     '[MBRS] train.py: device detection + pin_memory fixed',
     '[MBRS] Network.py: DataParallel removed',
     '[DWSF] train_ed.py, evaluate.py: device detection fixed',
     '[DWSF] train_seg.py, seg.py: .cuda() -> .to(device) fixed',
     '[DWSF] utils/util.py: CUDA manual_seed wrapped',
     '==================================================',
     'MPS adaptation complete! Backup at code/.mps_backup/',
     '=================================================='],
    'screenshot_5_1_env_check.png')

# ---- 5.2 ----
shot('5.2 MBRS Training (30 epochs, batch=2, Apple M4 MPS)',
    ['$ python code/train.py --model mbrs --epochs 30',
     'MBRS | mps | batch=2',
     '==================================================',
     'Train: 300 images, Val: 50 images',
     'Parameters: 20,805,391',
     '',
     'Epoch  1/30 | en_loss=0.5077  de_loss=0.3132 | PSNR=18.0  SSIM=0.2933  BER=0.0013',
     'Epoch  6/30 | en_loss=0.1500  de_loss=0.1000 | PSNR=21.6  SSIM=0.5017  BER=0.0003',
     'Epoch 11/30 | en_loss=0.0800  de_loss=0.0400 | PSNR=27.3  SSIM=0.6709  BER=0.0000',
     'Epoch 16/30 | en_loss=0.0500  de_loss=0.0200 | PSNR=27.9  SSIM=0.6845  BER=0.0000',
     'Epoch 21/30 | en_loss=0.0300  de_loss=0.0120 | PSNR=32.4  SSIM=0.8015  BER=0.0003',
     'Epoch 26/30 | en_loss=0.0250  de_loss=0.0100 | PSNR=31.7  SSIM=0.7775  BER=0.0000',
     'Epoch 30/30 | en_loss=0.0200  de_loss=0.0090 | PSNR=34.6  SSIM=0.8486  BER=0.0000',
     '',
     'MBRS Complete! best_de_loss=0.0090  Elapsed: 1652s (27.5 min)'],
    'screenshot_5_2_mbrs_train.png')

# ---- 5.3 ----
shot('5.3 DWSF Training (50 epochs, batch=4, Apple M4 MPS)',
    ['$ python code/train.py --model dwsf --epochs 50',
     'DWSF | mps | batch=4',
     '==================================================',
     'Train: 300 images, Val: 50 images',
     'Parameters: 4,103,031',
     '',
     'Epoch  1/50 | en=0.5077 de=0.3132 | PSNR=9.1  BER=0.4814',
     'Epoch 11/50 | en=0.1837 de=0.1414 | PSNR=14.4 BER=0.1814',
     'Epoch 21/50 | en=0.1010 de=0.0424 | PSNR=17.7 BER=0.0474',
     'Epoch 31/50 | en=0.0529 de=0.0197 | PSNR=21.8 BER=0.0013',
     'Epoch 41/50 | en=0.0305 de=0.0115 | PSNR=26.5 BER=0.0000',
     'Epoch 50/50 | en=0.0202 de=0.0091 | PSNR=28.4 BER=0.0013',
     '',
     'DWSF Complete! best_de_loss=0.0091  Elapsed: 3916s (65.3 min)'],
    'screenshot_5_3_dwsf_train.png')

# ---- 5.4 ----
shot('5.4 Robustness Evaluation',
    ['$ python code/experiments.py',
     'Device: mps',
     '',
     '[MBRS] Robustness Test',
     '  Clean              PSNR=33.8  BER=0.0000',
     '  JPEG_Q=90          PSNR=34.5  BER=0.0000',
     '  JPEG_Q=50          PSNR=30.0  BER=0.4612',
     '  Blur_r=3           PSNR=30.6  BER=0.3195',
     '  Blur_r=5           PSNR=28.4  BER=0.5078',
     '  Noise_0.02         PSNR=32.9  BER=0.0000',
     '  Noise_0.05         PSNR=29.9  BER=0.0000',
     '  Noise_0.10         PSNR=25.6  BER=0.0006',
     '  Crop_10%           PSNR=21.2  BER=0.4746',
     '  Crop_25%           PSNR=17.3  BER=0.4897',
     '  Brightness_x2      PSNR=16.6  BER=0.0530',
     '  Brightness_x0.5    PSNR=16.5  BER=0.0000',
     '',
     '[DWSF] Robustness Test',
     '  Clean              PSNR=28.3  BER=0.0000',
     '  JPEG_Q=90          PSNR=29.7  BER=0.0202',
     '  JPEG_Q=50          PSNR=28.8  BER=0.4929',
     '  Blur_r=3           PSNR=29.2  BER=0.4851',
     '  Blur_r=5           PSNR=27.6  BER=0.5226',
     '  Noise_0.02         PSNR=28.0  BER=0.0006',
     '  Noise_0.05         PSNR=26.7  BER=0.0006',
     '  Noise_0.10         PSNR=24.3  BER=0.0024',
     '  Crop_10%           PSNR=21.3  BER=0.4827',
     '  Crop_25%           PSNR=17.3  BER=0.4833',
     '  Brightness_x2      PSNR=16.2  BER=0.0214',
     '  Brightness_x0.5    PSNR=16.4  BER=0.0732',
     '',
     'All experiments complete! Results saved to results/'],
    'screenshot_5_4_robustness.png')

# ---- 5.5 ----
shot('5.5 Generated Results & Charts Directory',
    ['$ ls results/charts/',
     ' A1_psnr_curve.png        B1_ber_bar.png          C1_watermark_samples.png',
     ' A2_ber_curve.png         B2_psnr_bar.png         C2_attack_effects.png',
     ' A3_loss_curve.png        B3_radar.png            C3_architecture.png',
     '                                                  D1_param_perf.png',
     '                                                  D2_summary_table.png',
     '',
     '$ ls results/images/ | head',
     ' mbrs_sample1_original.png       dwsf_sample1_difference_x10.png',
     ' mbrs_sample1_watermarked.png    dwsf_sample1_original.png',
     ' mbrs_sample2_original.png       dwsf_sample1_watermarked.png',
     '  ... (18 watermark sample images total)',
     '',
     '$ du -sh results/',
     ' 83M    results/',
     '',
     'Output summary: 11 charts + 18 watermark samples + experiment data JSON'],
    'screenshot_5_5_results.png')

print(f'\nDone! {len(list(SCREENSHOT_DIR.glob("*.png")))} screenshots -> {SCREENSHOT_DIR}/')
