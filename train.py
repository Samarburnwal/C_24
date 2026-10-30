"""Train SRCNN or EDSR on degraded/clean pairs.

Example:
    python train.py --model edsr --train_dir data/train --val_dir data/val --epochs 50
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sr.data import SRPairs
from sr.models import build_model


def psnr_t(a, b):
    mse = torch.mean((a - b) ** 2, dim=(1, 2, 3)).clamp_min(1e-10)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="edsr", choices=["srcnn", "edsr"])
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--patch", type=int, default=96, help="HR patch size")
    ap.add_argument("--strength", default="medium", choices=["light", "medium", "heavy"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--repeat", type=int, default=10, help="crops per image per epoch")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    os.makedirs(args.out, exist_ok=True)

    tr = SRPairs(args.train_dir, args.scale, args.patch, args.strength, True, args.repeat)
    va = SRPairs(args.val_dir, args.scale, args.patch, args.strength, False)
    tl = DataLoader(tr, args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    vl = DataLoader(va, 8, num_workers=0)

    model = build_model(args.model, args.scale).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.L1Loss()
    best = -1.0
    ckpt = os.path.join(args.out, f"{args.model}_x{args.scale}.pt")

    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for lr_img, hr_img in tqdm(tl, desc=f"epoch {ep}/{args.epochs}", leave=False):
            lr_img, hr_img = lr_img.to(dev), hr_img.to(dev)
            loss = loss_fn(model(lr_img), hr_img)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item()
        sched.step()

        model.eval()
        scores = []
        with torch.no_grad():
            for lr_img, hr_img in vl:
                scores.append(psnr_t(model(lr_img.to(dev)), hr_img.to(dev)))
        val = sum(scores) / len(scores)
        print(f"epoch {ep:3d}  train L1 {run / len(tl):.4f}  val PSNR {val:.2f} dB")
        if val > best:
            best = val
            torch.save({"model": args.model, "scale": args.scale, "state": model.state_dict()}, ckpt)
    print(f"best val PSNR {best:.2f} dB -> {ckpt}")


if __name__ == "__main__":
    main()
