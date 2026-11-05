"""Inference-only model wrapper shared by the web app and the scripts.
Architecture and pre/post-processing match train/train.py."""
import contextlib
import os

import cv2
import numpy as np
import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    """Same 8-level U-Net as train/train.py."""

    def __init__(self, in_ch=1, out_ch=3, ngf=64, bottleneck_bn=False):
        super().__init__()
        e = [ngf, ngf * 2, ngf * 4, ngf * 8, ngf * 8, ngf * 8, ngf * 8, ngf * 8]
        self.enc = nn.ModuleList()
        prev = in_ch
        for i, c in enumerate(e):
            bn = i > 0 and (i < 7 or bottleneck_bn)
            layers = [nn.Conv2d(prev, c, 4, 2, 1, bias=not bn)] + ([nn.BatchNorm2d(c)] if bn else [])
            self.enc.append(nn.Sequential(*layers, nn.LeakyReLU(0.2)))
            prev = c
        dec = [(e[7], ngf * 8, True), (ngf * 16, ngf * 8, True), (ngf * 16, ngf * 8, True),
               (ngf * 16, ngf * 8, False), (ngf * 16, ngf * 4, False), (ngf * 8, ngf * 2, False),
               (ngf * 4, ngf, False)]
        self.dec = nn.ModuleList()
        for cin, cout, drop in dec:
            layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout)]
            self.dec.append(nn.Sequential(*layers, *([nn.Dropout(0.5)] if drop else []), nn.ReLU()))
        self.out = nn.Sequential(nn.ConvTranspose2d(ngf * 2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
        x = skips.pop()
        for blk in self.dec:
            x = torch.cat([blk(x), skips.pop()], 1)
        return self.out(x)


@contextlib.contextmanager
def test_mode(G, mode):
    """paper: dropout on + per-input BN stats (momentum 0); eval: G.eval()."""
    moms = {}
    if mode == "paper":
        G.train()
        for m in G.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                moms[m], m.momentum = m.momentum, 0.0
    else:
        G.eval()
    try:
        yield
    finally:
        for m, mo in moms.items():
            m.momentum = mo
        G.eval()


# generator.pth files written by an earlier (Turkish) version of train.py use other key names
_LEGACY = {"eskiz_kanal": "sketch_channels", "darbogaz_bn": "bottleneck_bn", "boyut": "size",
           "test_davranisi": "test_mode", "en_iyi_epoch": "best_epoch"}


def _load(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    meta = {_LEGACY.get(k, k): v for k, v in ck.items() if k != "state_dict"}
    if meta.get("test_mode") == "makale":
        meta["test_mode"] = "paper"
    sd = {("out." + k[4:] if k.startswith("son.") else k): v for k, v in ck["state_dict"].items()}
    return sd, meta


class Colorizer:
    """Loads generator.pth once; every request reuses the same model."""

    def __init__(self, weights, device=None, fp16=True, seed=42):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        sd, self.meta = _load(weights)
        self.size = int(self.meta.get("size", 512))
        self.sketch_channels = int(self.meta.get("sketch_channels", 1))
        self.mode = self.meta.get("test_mode", "paper")
        self.G = UNetGenerator(self.sketch_channels, 3, int(self.meta.get("ngf", 64)),
                               bool(self.meta.get("bottleneck_bn", 0)))
        self.G.load_state_dict(sd)
        self.G.to(self.device).eval()
        self.fp16 = bool(fp16) and self.device.type == "cuda"
        self.seed = seed
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    @staticmethod
    def split_pair(rgb):
        """Dataset pair (width = 2 x height): left = color, right = sketch."""
        w = rgb.shape[1] // 2
        return rgb[:, :w], rgb[:, w:]

    def preprocess(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
        s = self.size
        if gray.shape[:2] != (s, s):
            gray = cv2.resize(gray, (s, s), interpolation=cv2.INTER_AREA if max(gray.shape) > s else cv2.INTER_CUBIC)
        x = torch.from_numpy(np.ascontiguousarray(gray)).float().div(127.5).sub(1)[None, None]
        return x.repeat(1, 3, 1, 1) if self.sketch_channels == 3 else x

    @torch.no_grad()
    def forward(self, x):
        with test_mode(self.G, self.mode), \
                torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
            return self.G(x).float()

    @staticmethod
    def postprocess(y, size_hw=None):
        rgb = ((y[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        if size_hw and rgb.shape[:2] != size_hw:
            rgb = cv2.resize(rgb, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_CUBIC)
        return rgb

    def colorize(self, sketch_rgb, keep_size=True):
        """RGB sketch (numpy) -> RGB color image (numpy)."""
        torch.manual_seed(self.seed)  # deterministic dropout in "paper" mode
        y = self.forward(self.preprocess(sketch_rgb).to(self.device, non_blocking=True))
        return self.postprocess(y, sketch_rgb.shape[:2] if keep_size else None)


def read_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path, rgb):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
