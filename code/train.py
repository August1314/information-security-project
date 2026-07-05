#!/usr/bin/env python3
"""训练 MBRS 和 DWSF，保存结果到 results/experiments/"""

import sys, os, time, json, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MBRS'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DWSF'))

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import kornia

# === 配置 ===
H, W, BATCH = 128, 128, 4
DATA_DIR = str(Path(__file__).parent.parent / 'data')
SAVE_DIR = Path(__file__).parent.parent / 'results' / 'experiments'
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.files = list(Path(root).glob('*.jpg')) + list(Path(root).glob('*.png'))
        self.tf = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, i): return self.tf(Image.open(self.files[i]).convert('RGB'))

tf = transforms.Compose([
    transforms.RandomCrop((H, W), pad_if_needed=True, padding_mode='reflect'),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def train_one(model_type, epochs=50):
    """训练单个模型"""
    print(f'\n{"="*50}')
    print(f'训练 {model_type.upper()} | {DEVICE} | batch={BATCH}')
    print(f'{"="*50}')

    save_dir = SAVE_DIR / model_type
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    train_loader = DataLoader(ImgDataset(f'{DATA_DIR}/train', tf), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(ImgDataset(f'{DATA_DIR}/val', tf), batch_size=BATCH, shuffle=False)
    print(f'训练集: {len(train_loader.dataset)} 张, 验证集: {len(val_loader.dataset)} 张')

    # 构建模型
    if model_type == 'mbrs':
        from network.Encoder_MP_Decoder import EncoderDecoder
        from network.Discriminator import Discriminator
        noise = ['Combined([JpegMask(50),Jpeg(50),Identity()])']
        ed = EncoderDecoder(H, W, 64, noise).to(DEVICE)
        disc = Discriminator().to(DEVICE)
        msg_len = 64
    else:
        from networks.models.EncoderDecoder import EncoderDecoder
        from networks.models.Discriminator import Discriminator
        noise = ['Combined([Identity(),RandomJpegMask(50,100,padding=True),RandomJpeg(50,100,padding=True)])']
        ed = EncoderDecoder(H, W, 30, noise).to(DEVICE)
        disc = Discriminator().to(DEVICE)
        msg_len = 30

    params = sum(p.numel() for p in ed.parameters())
    print(f'参数: {params:,}')

    opt_ed = torch.optim.AdamW(ed.parameters(), lr=1e-4)
    opt_dis = torch.optim.AdamW(disc.parameters(), lr=1e-4)
    mse = torch.nn.MSELoss().to(DEVICE)
    bce = torch.nn.BCEWithLogitsLoss().to(DEVICE)

    history = []
    best_de = float('inf')
    t0 = time.time()

    for epoch in range(epochs):
        # ---- 训练 ----
        ed.encoder.train(); ed.decoder.train(); disc.train()
        en_sum = de_sum = d_sum = 0.0
        for images in train_loader:
            images = images.to(DEVICE); B = images.shape[0]
            ones = torch.ones(B, 1, device=DEVICE)
            zeros = torch.zeros(B, 1, device=DEVICE)
            msg = torch.randint(0, 2, (B, msg_len), device=DEVICE).float()

            encoded = ed.encoder(images, msg).clamp(-1, 1)
            noised = ed.noise([encoded, images])
            decoded = ed.decoder(noised)

            d_loss = bce(disc(images), ones) + bce(disc(encoded.detach()), zeros)
            opt_dis.zero_grad(); d_loss.backward(); opt_dis.step()

            en_loss = mse(images, encoded)
            de_loss = mse(decoded, msg)
            total = 1e-3 * bce(disc(encoded), ones) + 0.2 * en_loss + de_loss
            opt_ed.zero_grad(); total.backward(); opt_ed.step()

            en_sum += en_loss.item(); de_sum += de_loss.item()
            d_sum += d_loss.item()

        n = len(train_loader)
        en_avg, de_avg, d_avg = en_sum/n, de_sum/n, d_sum/n

        # ---- 验证 ----
        ed.encoder.eval(); ed.decoder.eval()
        psnr_sum = ssim_sum = ber_sum = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(DEVICE); B = images.shape[0]
                msg = torch.randint(0, 2, (B, msg_len), device=DEVICE).float()
                encoded = ed.encoder(images, msg).clamp(-1, 1)
                decoded = ed.decoder(encoded)
                psnr_sum += kornia.metrics.psnr(((encoded+1)/2).clamp(0,1), ((images+1)/2).clamp(0,1), 1).item()
                ssim_sum += kornia.metrics.ssim(((encoded+1)/2).clamp(0,1), ((images+1)/2).clamp(0,1), 11).mean().item()
                ber_sum += ((decoded>0.5).float() != msg).float().mean().item()
        m = len(val_loader)
        psnr, ssim, ber = psnr_sum/m, ssim_sum/m, ber_sum/m

        elapsed = time.time() - t0
        print(f'Epoch {epoch+1:3d}/{epochs} | en={en_avg:.4f} de={de_avg:.4f} '
              f'disc={d_avg:.4f} | PSNR={psnr:.1f} SSIM={ssim:.4f} BER={ber:.4f} | {elapsed:.0f}s')

        history.append({'epoch': epoch+1, 'en': en_avg, 'de': de_avg, 'disc': d_avg,
                        'psnr': psnr, 'ssim': ssim, 'ber': ber})

        # 保存最佳
        if de_avg < best_de:
            best_de = de_avg
            torch.save({'encoder': ed.encoder.state_dict(), 'decoder': ed.decoder.state_dict(),
                        'discriminator': disc.state_dict()}, save_dir / 'best.pth')

        # 每 10 epoch 保存检查点
        if (epoch+1) % 10 == 0:
            torch.save({'epoch': epoch+1, 'encoder': ed.encoder.state_dict(),
                        'decoder': ed.decoder.state_dict(),
                        'discriminator': disc.state_dict()}, save_dir / f'checkpoint_{epoch+1}.pth')

    # 保存历史
    torch.save({'history': history, 'params': params}, save_dir / 'history.pth')
    print(f'{model_type.upper()} 训练完成! 最佳 de_loss={best_de:.4f}')

    # 释放 GPU 内存
    del ed, disc, opt_ed, opt_dis
    if hasattr(torch, 'mps'): torch.mps.empty_cache()
    return history


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='both', choices=['mbrs', 'dwsf', 'both'])
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    if args.model in ['mbrs', 'both']:
        train_one('mbrs', args.epochs)
    if args.model in ['dwsf', 'both']:
        train_one('dwsf', args.epochs)
