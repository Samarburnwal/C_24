"""Upscale a single image or every frame of a video.

    python upscale.py --input frame.png --output frame_sr.png --model edsr --ckpt checkpoints/edsr_x4.pt
    python upscale.py --input cctv.mp4 --output cctv_sr.mp4 --model bicubic
    python upscale.py --input frame.png --output out.png --model realesrgan
"""
import argparse
import os

import cv2
import torch

from sr.data import to_bgr, to_tensor
from sr.models import Bicubic, build_model


def get_upscaler(name, ckpt, scale, dev):
    if name == "realesrgan":
        from sr.realesrgan_wrapper import RealESRGANUpscaler
        return RealESRGANUpscaler(scale, dev).upscale
    if name == "bicubic":
        model = Bicubic(scale).to(dev).eval()
    else:
        if not ckpt:
            raise SystemExit(f"--ckpt is required for model '{name}'")
        ck = torch.load(ckpt, map_location=dev)
        model = build_model(ck["model"], ck["scale"]).to(dev)
        model.load_state_dict(ck["state"])
        model.eval()

    @torch.no_grad()
    def fn(bgr):
        return to_bgr(model(to_tensor(bgr).unsqueeze(0).to(dev))[0])
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="bicubic", choices=["bicubic", "srcnn", "edsr", "realesrgan"])
    ap.add_argument("--ckpt")
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    up = get_upscaler(args.model, args.ckpt, args.scale, dev)

    if os.path.splitext(args.input)[1].lower() in (".mp4", ".avi", ".mov", ".mkv"):
        cap = cv2.VideoCapture(args.input)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer, n = None, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            sr = up(frame)
            if writer is None:
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (sr.shape[1], sr.shape[0]))
            writer.write(sr)
            n += 1
        cap.release()
        if writer:
            writer.release()
        print(f"{n} frames -> {args.output}")
    else:
        img = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Cannot read {args.input}")
        cv2.imwrite(args.output, up(img))
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
