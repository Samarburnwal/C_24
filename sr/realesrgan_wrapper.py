"""Optional wrapper around the pretrained Real-ESRGAN x4 model.

Install:  pip install realesrgan basicsr
Weights are downloaded automatically on first use (needs internet) from the
official Real-ESRGAN GitHub release (RealESRGAN_x4plus.pth).

NOTE: Real-ESRGAN is GAN-based. It produces sharp, plausible detail that is
*invented*, not recovered - see the README section on forensic limitations.
"""
import os

import cv2

URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


def _patch_torchvision():
    """basicsr 1.4.2 imports torchvision.transforms.functional_tensor, which was
    removed in torchvision >= 0.17. Provide it as an alias so basicsr imports."""
    import sys
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ModuleNotFoundError:
        import types
        from torchvision.transforms import functional as F
        shim = types.ModuleType("torchvision.transforms.functional_tensor")
        shim.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = shim


class RealESRGANUpscaler:
    def __init__(self, scale=4, device="cpu", weights_dir="weights"):
        if scale != 4:
            raise ValueError("The bundled Real-ESRGAN model is x4 only.")
        _patch_torchvision()
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from basicsr.utils.download_util import load_file_from_url
            from realesrgan import RealESRGANer
        except ImportError as e:
            raise ImportError(
                f"Could not import Real-ESRGAN ({e}). "
                "If the package is missing: pip install realesrgan basicsr") from e

        os.makedirs(weights_dir, exist_ok=True)
        path = load_file_from_url(URL, model_dir=weights_dir, progress=True)
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        half = device == "cuda"
        self.up = RealESRGANer(scale=4, model_path=path, model=net, tile=256 if half else 128,
                               tile_pad=10, pre_pad=0, half=half,
                               device=None if device == "cuda" else "cpu")

    def upscale(self, bgr):
        out, _ = self.up.enhance(bgr, outscale=4)
        return out