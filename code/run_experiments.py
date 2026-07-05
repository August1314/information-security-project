#!/usr/bin/env python3
"""
统一实验脚本：训练、评估、对比 MBRS vs DWSF

用法:
    # 准备数据并训练
    python code/run_experiments.py --mode train --epochs 50 --data_dir /path/to/coco

    # 仅评估（需要已有模型）
    python code/run_experiments.py --mode eval --data_dir /path/to/coco

    # 完整流程
    python code/run_experiments.py --mode all --epochs 50 --data_dir /path/to/coco
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / 'MBRS'))
sys.path.insert(0, str(BASE / 'DWSF'))

import kornia
from network.Encoder_MP_Decoder import EncoderDecoder as MBRS_EncoderDecoder
from network.Discriminator import Discriminator as MBRS_Discriminator
from network.Noise import Noise as MBRS_Noise
from networks.models.EncoderDecoder import EncoderDecoder as DWSF_EncoderDecoder
from networks.models.Discriminator import Discriminator as DWSF_Discriminator
from networks.models.Noiser import Noise as DWSF_Noise

# ============= 配置 =============
H, W = 128, 128
BATCH_SIZE = 16
MBRS_MSG_LEN = 64
DWSF_MSG_LEN = 30
SAVE_DIR = BASE.parent / 'results' / 'experiments'

# MPS 优先
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


class ImageDataset(torch.utils.data.Dataset):
    """从目录加载图像的数据集"""
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.files = list(self.root.rglob('*.jpg')) + list(self.root.rglob('*.png'))
        self.transform = transform or transforms.Compose([
            transforms.RandomCrop((H, W), pad_if_needed=True, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img)


class Trainer:
    """统一的训练器"""
    def __init__(self, model_type='mbrs'):
        self.model_type = model_type
        self.save_dir = SAVE_DIR / model_type
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def build_model(self):
        if self.model_type == 'mbrs':
            noise = ['Combined([JpegMask(50),Jpeg(50),Identity()])']
            ed = MBRS_EncoderDecoder(H, W, MBRS_MSG_LEN, noise).to(DEVICE)
            disc = MBRS_Discriminator().to(DEVICE)
            return ed, disc
        else:
            noise = ['Combined([Identity(),RandomJpegMask(50,100,padding=True),RandomJpeg(50,100,padding=True)])']
            ed = DWSF_EncoderDecoder(H, W, DWSF_MSG_LEN, noise).to(DEVICE)
            disc = DWSF_Discriminator().to(DEVICE)
            return ed, disc

    def train_epoch(self, ed, disc, loader, optimizers, epoch):
        ed.encoder.train()
        ed.decoder.train()
        disc.train()
        opt_ed, opt_disc = optimizers

        mse = nn.MSELoss().to(DEVICE)
        bce = nn.BCEWithLogitsLoss().to(DEVICE)
        msg_len = MBRS_MSG_LEN if self.model_type == 'mbrs' else DWSF_MSG_LEN

        losses = {'en': [], 'de': [], 'disc': []}
        t0 = time.time()

        for images in loader:
            images = images.to(DEVICE)
            B = images.shape[0]
            ones = torch.ones(B, 1, device=DEVICE)
            zeros = torch.zeros(B, 1, device=DEVICE)
            msg = torch.randint(0, 2, (B, msg_len), device=DEVICE).float()

            encoded = ed.encoder(images, msg).clamp(-1, 1)
            noised = ed.noise([encoded, images])
            decoded = ed.decoder(noised)

            # Discriminator
            d_loss = bce(disc(images), ones) + bce(disc(encoded.detach()), zeros)
            opt_disc.zero_grad()
            d_loss.backward()
            opt_disc.step()

            # Encoder-Decoder
            g_loss = bce(disc(encoded), ones)
            en_loss = mse(images, encoded)
            de_loss = mse(decoded, msg)
            total = 1e-3 * g_loss + 0.2 * en_loss + de_loss
            opt_ed.zero_grad()
            total.backward()
            opt_ed.step()

            losses['en'].append(en_loss.item())
            losses['de'].append(de_loss.item())
            losses['disc'].append(d_loss.item())

        elapsed = time.time() - t0
        return {k: np.mean(v) for k, v in losses.items()}, elapsed

    @torch.no_grad()
    def evaluate(self, ed, loader, noise_layer=None):
        """评估：计算 PSNR、SSIM、BER"""
        ed.encoder.eval()
        ed.decoder.eval()
        msg_len = MBRS_MSG_LEN if self.model_type == 'mbrs' else DWSF_MSG_LEN

        if noise_layer:
            val_noise = MBRS_Noise(noise_layer).to(DEVICE) if self.model_type == 'mbrs' \
                else DWSF_Noise(noise_layer).to(DEVICE)
        else:
            val_noise = None

        psnr_list, ssim_list, ber_list = [], [], []

        for images in loader:
            images = images.to(DEVICE)
            B = images.shape[0]
            msg = torch.randint(0, 2, (B, msg_len), device=DEVICE).float()

            encoded = ed.encoder(images, msg).clamp(-1, 1)
            noised = val_noise([encoded, images]) if val_noise else encoded
            decoded = ed.decoder(noised)

            # PSNR
            psnr = kornia.metrics.psnr(
                ((encoded + 1) / 2).clamp(0, 1),
                ((images + 1) / 2).clamp(0, 1), 1)
            psnr_list.append(psnr.item())

            # SSIM
            ssim = kornia.metrics.ssim(
                ((encoded + 1) / 2).clamp(0, 1),
                ((images + 1) / 2).clamp(0, 1), window_size=11).mean()
            ssim_list.append(ssim.item())

            # BER
            pred = (decoded > 0.5).float()
            ber = (pred != msg).float().mean()
            ber_list.append(ber.item())

        return {
            'psnr': np.mean(psnr_list),
            'ssim': np.mean(ssim_list),
            'ber': np.mean(ber_list),
        }

    def train(self, train_dir, epochs=50):
        """完整训练流程"""
        print(f'\n{"="*50}')
        print(f'训练 {self.model_type.upper()} | 设备: {DEVICE}')
        print(f'{"="*50}')

        transform = transforms.Compose([
            transforms.RandomCrop((H, W), pad_if_needed=True, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        dataset = ImageDataset(train_dir, transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f'  数据: {len(dataset)} 张图像, {len(loader)} batches')

        ed, disc = self.build_model()
        n_params = sum(p.numel() for p in ed.parameters())
        print(f'  参数: {n_params:,}')

        opt_ed = torch.optim.AdamW(ed.parameters(), lr=1e-4)
        opt_disc = torch.optim.AdamW(disc.parameters(), lr=1e-4)

        best_loss = float('inf')
        history = []

        for epoch in range(epochs):
            losses, elapsed = self.train_epoch(ed, disc, loader, (opt_ed, opt_disc), epoch)

            # 每 10 轮评估
            if epoch % 10 == 0 or epoch == epochs - 1:
                metrics = self.evaluate(ed, loader)
                print(f'  Epoch {epoch:3d} | en={losses["en"]:.4f} de={losses["de"]:.4f} '
                      f'disc={losses["disc"]:.4f} | PSNR={metrics["psnr"]:.1f} '
                      f'SSIM={metrics["ssim"]:.4f} BER={metrics["ber"]:.4f} | {elapsed:.1f}s')

            entry = {'epoch': epoch, **losses}
            if 'metrics' in dir():
                entry.update(metrics)
            history.append(entry)

            if losses['de'] < best_loss:
                best_loss = losses['de']
                torch.save({
                    'encoder': ed.encoder.state_dict(),
                    'decoder': ed.decoder.state_dict(),
                    'discriminator': disc.state_dict(),
                }, self.save_dir / 'best.pth')

        # 保存训练历史
        torch.save({'history': history, 'model_type': self.model_type},
                   self.save_dir / 'history.pth')
        return history

    def load_best(self):
        """加载最佳模型"""
        ed, disc = self.build_model()
        ckpt = torch.load(self.save_dir / 'best.pth', map_location=DEVICE)
        ed.encoder.load_state_dict(ckpt['encoder'])
        ed.decoder.load_state_dict(ckpt['decoder'])
        disc.load_state_dict(ckpt['discriminator'])
        return ed, disc


def compare_models(train_dir, attack_types=None):
    """对比两个模型在各种攻击下的表现"""
    if attack_types is None:
        attack_types = {
            'Clean': [],
            'JPEG Q=50': ['JpegMask(50)', 'Jpeg(50)'],
            'JPEG Q=80': ['JpegMask(80)', 'Jpeg(80)'],
            'Gaussian Blur': ['GaussianFilter(5)'],
            'Gaussian Noise': ['GaussianNoise(0,0.1)'],
            'Salt & Pepper': ['SaltPepper(0.1)'],
            'Crop 10%': ['Crop(0.1)'],
            'Combined': ['Combined([JpegMask(50),Jpeg(50),GaussianFilter(5),Crop(0.1)])'],
        }

    transform = transforms.Compose([
        transforms.CenterCrop((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = ImageDataset(train_dir, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    results = {}
    print(f'\n{"="*60}')
    print(f'鲁棒性对比: MBRS vs DWSF')
    print(f'{"="*60}')

    for model_name in ['mbrs', 'dwsf']:
        trainer = Trainer(model_name)
        try:
            ed, disc = trainer.load_best()
        except FileNotFoundError:
            print(f'  {model_name}: 未找到预训练模型，跳过')
            continue

        model_results = {}
        for attack_name, noise_layers in attack_types.items():
            if not noise_layers:
                metrics = trainer.evaluate(ed, loader)
            else:
                metrics = trainer.evaluate(ed, loader, noise_layers)
            model_results[attack_name] = metrics

        results[model_name] = model_results

        print(f'\n  {model_name.upper()}:')
        print(f'  {"攻击类型":<20} {"PSNR":>8} {"SSIM":>8} {"BER":>8}')
        print(f'  {"-"*44}')
        for attack_name, m in model_results.items():
            print(f'  {attack_name:<20} {m["psnr"]:>8.2f} {m["ssim"]:>8.4f} {m["ber"]:>8.4f}')

    # 保存对比结果
    with open(SAVE_DIR / 'comparison.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='all', choices=['train', 'eval', 'all'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    if args.mode in ['train', 'all']:
        for model_type in ['mbrs', 'dwsf']:
            Trainer(model_type).train(args.data_dir, epochs=args.epochs)

    if args.mode in ['eval', 'all']:
        compare_models(args.data_dir)
