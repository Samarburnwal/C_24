"""Compare models on a folder of HR test images (degraded on the fly, fixed seed).

Example:
    python evaluate.py --test_dir data/test --ckpt checkpoints/srcnn_x4.pt checkpoints/edsr_x4.pt \
        --realesrgan --save_dir results
Writes results/metrics.csv and side-by-side comparison images.
"""
import argparse
import csv
import os
import random

import cv2
import numpy as np
import torch

from sr.data import list_images, to_bgr, to_tensor
from sr.degrade import degrade
from sr.metrics import all_metrics
from sr.models import Bicubic, build_model


def load_ckpt(path, dev):
    ck = torch.load(path, map_location=dev)
    m = build_model(ck["model"], ck["scale"]).to(dev)
    m.load_state_dict(ck["state"])
    return ck["model"].upper(), m.eval()


@torch.no_grad()
def run(model, lr_bgr, dev):
    out = model(to_tensor(lr_bgr).unsqueeze(0).to(dev))[0]
    return to_bgr(out)


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--strength", default="medium", choices=["light", "medium", "heavy"])
    ap.add_argument("--max_side", type=int, default=256, help="center-crop HR test images to this size")
    ap.add_argument("--realesrgan", action="store_true", help="also evaluate pretrained Real-ESRGAN")
    ap.add_argument("--save_dir", default="results")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    models = {"Bicubic": Bicubic(args.scale).to(dev).eval()}
    for c in args.ckpt:
        name, m = load_ckpt(c, dev)
        models[name] = m
    esr = None
    if args.realesrgan:
        from sr.realesrgan_wrapper import RealESRGANUpscaler
        esr = RealESRGANUpscaler(args.scale, dev)

    rows, agg = [], {}
    files = list_images(args.test_dir)
    for i, f in enumerate(files):
        hr = cv2.imread(f, cv2.IMREAD_COLOR)
        s = args.max_side - args.max_side % args.scale
        h, w = hr.shape[:2]
        y, x = max((h - s) // 2, 0), max((w - s) // 2, 0)
        hr = hr[y:y + s, x:x + s]
        h, w = hr.shape[0] - hr.shape[0] % args.scale, hr.shape[1] - hr.shape[1] % args.scale
        hr = hr[:h, :w]

        np.random.seed(i)
        lr = degrade(hr, args.scale, args.strength, random.Random(i))
        panels = [label(cv2.resize(lr, (w, h), interpolation=cv2.INTER_NEAREST), "LOW-RES input")]
        outputs = {n: run(m, lr, dev) for n, m in models.items()}
        if esr is not None:
            outputs["RealESRGAN"] = esr.upscale(lr)
        for n, sr in outputs.items():
            if sr.shape != hr.shape:
                sr = cv2.resize(sr, (w, h), interpolation=cv2.INTER_CUBIC)
            m = all_metrics(sr, hr, dev)
            rows.append({"image": os.path.basename(f), "model": n, **m})
            agg.setdefault(n, []).append(m)
            panels.append(label(sr, f"{n}  {m['psnr']:.2f}dB"))
        panels.append(label(hr, "GROUND TRUTH"))
        cv2.imwrite(os.path.join(args.save_dir, f"compare_{i:03d}.png"), np.hstack(panels))

    with open(os.path.join(args.save_dir, "metrics.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["image", "model", "psnr", "ssim", "lpips"])
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n{'Model':<12}{'PSNR (dB)':>10}{'SSIM':>9}{'LPIPS':>9}   (n={len(files)})")
    summary = []
    for n, ms in agg.items():
        p = np.mean([m["psnr"] for m in ms]); s_ = np.mean([m["ssim"] for m in ms])
        l = [m["lpips"] for m in ms if m["lpips"] is not None]
        l = np.mean(l) if l else float("nan")
        print(f"{n:<12}{p:>10.2f}{s_:>9.4f}{l:>9.4f}")
        summary.append((n, p, s_, l))
    with open(os.path.join(args.save_dir, "summary.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["model", "psnr", "ssim", "lpips"])
        wr.writerows(summary)


if __name__ == "__main__":
    main()
