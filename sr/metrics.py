"""Image-quality metrics: PSNR, SSIM (fidelity) and LPIPS (perceptual)."""
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_lpips_net = None


def psnr(a_bgr, b_bgr):
    return float(peak_signal_noise_ratio(a_bgr, b_bgr, data_range=255))


def ssim(a_bgr, b_bgr):
    return float(structural_similarity(a_bgr, b_bgr, channel_axis=2, data_range=255))


def lpips_score(a_bgr, b_bgr, device="cpu"):
    """Lower is better. Returns None if the ``lpips`` package is unavailable."""
    global _lpips_net
    if _lpips_net is False:  # previously failed to load; don't retry every image
        return None
    try:
        import lpips
        if _lpips_net is None:
            _lpips_net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception as e:  # missing package or no internet for pretrained weights
        print(f"[warn] LPIPS unavailable ({type(e).__name__}); reporting PSNR/SSIM only.")
        _lpips_net = False
        return None

    def prep(x):
        t = torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1).float() / 127.5 - 1
        return t.unsqueeze(0).to(device)

    with torch.no_grad():
        return float(_lpips_net(prep(a_bgr), prep(b_bgr)).item())


def all_metrics(sr_bgr, hr_bgr, device="cpu"):
    return {"psnr": psnr(sr_bgr, hr_bgr), "ssim": ssim(sr_bgr, hr_bgr),
            "lpips": lpips_score(sr_bgr, hr_bgr, device)}
