"""Super-resolution models: Bicubic baseline, SRCNN, and a compact EDSR."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bicubic(nn.Module):
    """Non-learned baseline: bicubic interpolation."""

    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic",
                             align_corners=False).clamp(0, 1)


class SRCNN(nn.Module):
    """Dong et al. 2014. Operates on a bicubic-upscaled input (pre-upsampling)."""

    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4), nn.ReLU(True),
            nn.Conv2d(64, 32, 5, padding=2), nn.ReLU(True),
            nn.Conv2d(32, 3, 5, padding=2),
        )

    def forward(self, x):
        up = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return (up + self.body(up)).clamp(0, 1)  # residual on top of bicubic


class ResBlock(nn.Module):
    def __init__(self, c, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True),
                                  nn.Conv2d(c, c, 3, padding=1))
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class EDSR(nn.Module):
    """Lim et al. 2017 (no batch-norm, residual scaling, sub-pixel upsampling).

    Defaults (16 blocks / 64 ch) are a lightweight variant that trains on a free
    Colab GPU. Use n_blocks=32, n_feats=256 for the paper-size model.
    """

    def __init__(self, scale=4, n_blocks=16, n_feats=64):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_blocks)],
                                  nn.Conv2d(n_feats, n_feats, 3, padding=1))
        up = []
        for _ in range({2: 1, 4: 2, 8: 3}[scale]):
            up += [nn.Conv2d(n_feats, n_feats * 4, 3, padding=1), nn.PixelShuffle(2)]
        self.up = nn.Sequential(*up)
        self.tail = nn.Conv2d(n_feats, 3, 3, padding=1)

    def forward(self, x):
        # Deviation from the original paper: a bicubic skip connection, so the
        # network only learns the missing high-frequency residual. This starts
        # training at bicubic quality and converges much faster on short runs.
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        h = self.head(x)
        h = h + self.body(h)
        return (base + self.tail(self.up(h))).clamp(0, 1)


def build_model(name, scale=4):
    name = name.lower()
    if name == "bicubic":
        return Bicubic(scale)
    if name == "srcnn":
        return SRCNN(scale)
    if name == "edsr":
        return EDSR(scale)
    raise ValueError(f"Unknown model '{name}'. Choose from: bicubic, srcnn, edsr")
