#!/usr/bin/env python
"""pix2pix (Isola et al., 2017) for anime sketch colorization. Intended for Kaggle (Tesla T4).

    pip install -q clean-fid
    python train.py --mode smoke      # tiny end-to-end check (train -> ckpt -> resume -> eval)
    python train.py --mode benchmark  # epoch time + GPU memory estimate
    python train.py --mode train      # full training (batch 4, 25 epochs, resumable)
    python train.py --mode evaluate   # test metrics from a saved checkpoint
"""
import argparse
import contextlib
import csv
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


# ----------------------------------------------------------------------------- config
def parse_args():
    p = argparse.ArgumentParser(description="pix2pix sketch colorization")
    p.add_argument("--mode", required=True, choices=["smoke", "benchmark", "train", "evaluate"])
    p.add_argument("--data_root", default="/kaggle/input")
    p.add_argument("--out", default=None, help="default /kaggle/working (smoke: /kaggle/working/smoke)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_local", type=int, default=0, help="allow running outside Kaggle")

    # data
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--color_side", default="left", choices=["left", "right"],
                   help="expected color half of each pair; the script stops if the check disagrees")
    p.add_argument("--split", default="0.8,0.1,0.1", help="train,val,test")
    p.add_argument("--flip_p", type=float, default=0.5, help="joint horizontal flip")
    p.add_argument("--thicken_p", type=float, default=0.3, help="sketch-only line thickening")
    p.add_argument("--thicken_kernels", default="2,3")

    # model
    p.add_argument("--sketch_channels", type=int, default=1, choices=[1, 3])
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--bottleneck_bn", type=int, default=0)
    p.add_argument("--test_mode", default="auto", choices=["auto", "paper", "eval"],
                   help="paper: dropout on + per-input BN stats; eval: model.eval(); auto: pick by val L1")

    # loss
    p.add_argument("--use_gan", type=int, default=1)
    p.add_argument("--use_l1", type=int, default=1)
    p.add_argument("--use_perc", type=int, default=1)
    p.add_argument("--lambda_l1", type=float, default=100.0)
    p.add_argument("--lambda_perc", type=float, default=10.0)
    p.add_argument("--vgg_layer", default="relu4_4",
                   choices=["relu1_2", "relu2_2", "relu3_4", "relu4_4", "relu5_4"])

    # training
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--amp", type=int, default=1, help="fp16 autocast; disabled automatically on NaN")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--ckpt_every", type=int, default=1, help="epochs between last.pt saves")
    p.add_argument("--keep_every", type=int, default=5, help="epochs between persistent epoch_XXX.pt")
    p.add_argument("--keep_last", type=int, default=2, help="number of epoch_XXX.pt to keep")
    p.add_argument("--sample_every", type=int, default=5)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--val_n", type=int, default=500, help="validation images per check (0 = all)")
    p.add_argument("--select", default="l1", choices=["l1", "ssim"], help="best-model criterion")
    p.add_argument("--resume", default="auto", help="auto | none | path to last.pt")
    p.add_argument("--max_hours", type=float, default=11.0, help="stop before an epoch that would exceed this")

    # benchmark / smoke
    p.add_argument("--bench_batch", default="4")
    p.add_argument("--bench_steps", type=int, default=200)
    p.add_argument("--bench_warmup", type=int, default=20)
    p.add_argument("--smoke_n", type=int, default=64)

    # evaluation (fixed values)
    p.add_argument("--ckpt", default=None, help="evaluate: checkpoint or generator.pth")
    p.add_argument("--canny_low", type=int, default=100)
    p.add_argument("--canny_high", type=int, default=200)
    p.add_argument("--f1_tol", type=int, default=2, help="edge matching tolerance (px)")
    p.add_argument("--fid", type=int, default=1)
    return p.parse_args()


def log(*a):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(_):
    s = torch.initial_seed() % 2 ** 32
    random.seed(s)
    np.random.seed(s)


@contextlib.contextmanager
def fixed_rng(seed):
    """Deterministic evaluation without disturbing the training RNG state."""
    py, npy = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        seed_all(seed)
        try:
            yield
        finally:
            random.setstate(py)
            np.random.set_state(npy)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- data discovery and split
def find_data(root):
    """Shallowest train/ and val/ folders (the dataset also ships a data/data/ copy)."""
    found = {"train": [], "val": []}
    for d, _, files in os.walk(root):
        name = os.path.basename(d).lower()
        if name in found:
            n = sum(f.lower().endswith(IMG_EXT) for f in files)
            if n:
                found[name].append((d.count(os.sep), -n, d))
    for k in found:
        if not found[k]:
            sys.exit(f"ERROR: no '{k}' folder under {root}")
        found[k].sort()
    tr, va = found["train"][0][2], found["val"][0][2]
    base = os.path.commonpath([tr, va])
    files = [os.path.relpath(os.path.join(d, f), base).replace("\\", "/")
             for d in (tr, va) for f in sorted(os.listdir(d)) if f.lower().endswith(IMG_EXT)]
    return base, files, tr, va


def check_halves(base, files, out, seed, expected, n=200):
    """The half with higher mean HSV saturation is the color image."""
    sample = random.Random(seed).sample(files, min(n, len(files)))
    s_left, s_right = [], []
    d = os.path.join(out, "half_check")
    os.makedirs(d, exist_ok=True)
    for i, rel in enumerate(sample):
        img = cv2.imread(os.path.join(base, rel), cv2.IMREAD_COLOR)
        w = img.shape[1] // 2
        a = float(cv2.cvtColor(img[:, :w], cv2.COLOR_BGR2HSV)[..., 1].mean())
        b = float(cv2.cvtColor(img[:, w:], cv2.COLOR_BGR2HSV)[..., 1].mean())
        s_left.append(a)
        s_right.append(b)
        if i < 4:
            cv2.imwrite(os.path.join(d, f"sample{i}_left_sat{a:.1f}.png"), img[:, :w])
            cv2.imwrite(os.path.join(d, f"sample{i}_right_sat{b:.1f}.png"), img[:, w:])
    color = "left" if np.mean(s_left) > np.mean(s_right) else "right"
    res = {"samples": len(sample), "mean_sat_left": round(float(np.mean(s_left)), 2),
           "mean_sat_right": round(float(np.mean(s_right)), 2), "color_side": color,
           "sketch_side": "right" if color == "left" else "left", "expected_color_side": expected}
    write_json(os.path.join(out, "half_check.json"), res)
    log(f"HALF CHECK: saturation left={res['mean_sat_left']} right={res['mean_sat_right']} -> color={color}")
    if color != expected:
        sys.exit(f"ERROR: color half is '{color}', expected '{expected}'.")
    return res


def make_split(files, ratios, seed, out, base):
    """train+val merged, shuffled with a fixed seed, split 80/10/10; saved to split.json."""
    path = os.path.join(out, "split.json")
    r = [float(v) for v in ratios.split(",")]
    if os.path.exists(path):
        s = read_json(path)
        if s["seed"] == seed and s["ratios"] == r and s["total"] == len(files):
            return s
    lst = sorted(files)
    random.Random(seed).shuffle(lst)
    n = len(lst)
    n_tr, n_va = int(round(r[0] * n)), int(round(r[1] * n))
    s = {"seed": seed, "ratios": r, "total": n, "root": base,
         "counts": {"train": n_tr, "val": n_va, "test": n - n_tr - n_va},
         "train": lst[:n_tr], "val": lst[n_tr:n_tr + n_va], "test": lst[n_tr + n_va:]}
    write_json(path, s)
    return s


# ----------------------------------------------------------------------------- dataset
class PairDataset(Dataset):
    """Left half = color target, right half = sketch (1-channel gray). Values in [-1, 1]."""

    def __init__(self, base, paths, a, augment):
        self.base, self.paths, self.a, self.augment = base, paths, a, augment
        self.kernels = [int(k) for k in a.thicken_kernels.split(",")]

    def __len__(self):
        return len(self.paths)

    def _resize(self, im):
        s = self.a.size
        if im.shape[:2] == (s, s):
            return im
        return cv2.resize(im, (s, s), interpolation=cv2.INTER_AREA if im.shape[0] > s else cv2.INTER_CUBIC)

    def __getitem__(self, i):
        path = os.path.join(self.base, self.paths[i])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cannot read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        w = img.shape[1] // 2
        left, right = img[:, :w], img[:, w:]
        color, sketch = (left, right) if self.a.color_side == "left" else (right, left)
        color = self._resize(color)
        sketch = cv2.cvtColor(self._resize(sketch), cv2.COLOR_RGB2GRAY)
        if self.augment:
            if random.random() < self.a.flip_p:          # same flip for both
                color, sketch = color[:, ::-1], sketch[:, ::-1]
            if random.random() < self.a.thicken_p:       # erode = thicker dark lines
                k = random.choice(self.kernels)
                sketch = cv2.erode(np.ascontiguousarray(sketch), np.ones((k, k), np.uint8))
        x = torch.from_numpy(np.ascontiguousarray(sketch)).float().div(127.5).sub(1).unsqueeze(0)
        if self.a.sketch_channels == 3:
            x = x.repeat(3, 1, 1)
        y = torch.from_numpy(np.ascontiguousarray(color)).float().div(127.5).sub(1).permute(2, 0, 1)
        return x, y, i


def make_loader(ds, a, batch, shuffle, gen=None, drop_last=False):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=a.num_workers, pin_memory=True,
                      drop_last=drop_last, generator=gen, worker_init_fn=worker_init)


# ----------------------------------------------------------------------------- models
class UNetGenerator(nn.Module):
    """8-level U-Net. Encoder C64-C128-C256-C512x5 (4x4, stride 2, LeakyReLU 0.2, no BN on first layer).
    Decoder CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128 (ReLU, dropout 0.5 on first 3). Skips i <-> n-i."""

    def __init__(self, in_ch=1, out_ch=3, ngf=64, bottleneck_bn=False):
        super().__init__()
        e = [ngf, ngf * 2, ngf * 4, ngf * 8, ngf * 8, ngf * 8, ngf * 8, ngf * 8]
        self.enc = nn.ModuleList()
        prev = in_ch
        for i, c in enumerate(e):
            bn = i > 0 and (i < 7 or bottleneck_bn)  # errata: no BN at the bottleneck
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
        x = skips.pop()  # bottleneck (2x2 for 512 input)
        for blk in self.dec:
            x = torch.cat([blk(x), skips.pop()], 1)
        return self.out(x)


class PatchGAN70(nn.Module):
    """70x70 PatchGAN: C64-C128-C256-C512 -> 1 (logits). Strides 2,2,2,1,1."""

    def __init__(self, in_ch, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1))

    def forward(self, sketch, color):
        return self.net(torch.cat([sketch, color], 1))


def init_weights(m):
    """Conv weights N(0, 0.02); BN scale N(1, 0.02), shift 0."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class VGGPerceptual(nn.Module):
    """MSE between frozen ImageNet VGG-19 features; input mapped [-1,1] -> ImageNet normalization."""
    CUT = {"relu1_2": 4, "relu2_2": 9, "relu3_4": 18, "relu4_4": 27, "relu5_4": 36}

    def __init__(self, layer):
        super().__init__()
        from torchvision.models import vgg19
        try:
            from torchvision.models import VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        except ImportError:
            vgg = vgg19(pretrained=True)
        self.f = vgg.features[: self.CUT[layer]].eval()
        for p in self.f.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        return super().train(False)

    def _norm(self, x):
        return ((x + 1) / 2 - self.mean) / self.std

    def forward(self, fake, real):
        ff = self.f(self._norm(fake))
        with torch.no_grad():
            fr = self.f(self._norm(real))
        return F.mse_loss(ff.float(), fr.float())


@contextlib.contextmanager
def test_mode(G, mode):
    """paper: dropout on, BN uses current input stats (momentum 0 keeps running stats intact); eval: G.eval()."""
    was_training, moms = G.training, {}
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
        G.train(was_training)


def active_mode(a):
    return a.test_mode if a.test_mode in ("paper", "eval") else "eval"


# ----------------------------------------------------------------------------- metrics
def autocast(on):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=on)


def grad_scaler(on):
    try:
        return torch.amp.GradScaler("cuda", enabled=on)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=on)


def to_rgb8(t):
    return ((t.detach().float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


def to_gray8(t):
    return ((t[0].detach().float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()


def ssim_rgb(a, b):
    """SSIM on RGB in [0, 1], Gaussian window sigma 1.5 (Wang et al., 2004)."""
    from skimage.metrics import structural_similarity as ssim
    a, b = a.astype(np.float64) / 255, b.astype(np.float64) / 255
    kw = dict(data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    try:
        return float(ssim(a, b, channel_axis=2, **kw))
    except TypeError:
        return float(ssim(a, b, multichannel=True, **kw))


def edge_f1(sketch, rgb, low, high, tol):
    """F1 between Canny edges of the sketch (reference) and of the image, with +-tol px tolerance."""
    ref = cv2.Canny(sketch, low, high) > 0
    pred = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), low, high) > 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    ref_d = cv2.dilate(ref.astype(np.uint8), k) > 0
    pred_d = cv2.dilate(pred.astype(np.uint8), k) > 0
    p = float((pred & ref_d).sum()) / pred.sum() if pred.sum() else 0.0
    r = float((ref & pred_d).sum()) / ref.sum() if ref.sum() else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def summary(v):
    v = np.asarray(v, dtype=np.float64)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4), "n": int(v.size)}


@torch.no_grad()
def generate(G, x, mode, amp):
    with test_mode(G, mode), autocast(amp):
        return G(x).float()


@torch.no_grad()
def evaluate_set(G, base, paths, a, mode, amp, ssim=True, f1=True, fid_dir=None, tag=""):
    """Batch 1 (same as the web app): L1, SSIM, edge F1; optionally saves PNGs for FID."""
    dl = make_loader(PairDataset(base, paths, a, False), a, 1, False)
    r = {k: [] for k in ("ssim", "l1", "f1", "p", "r", "ref_f1", "ref_p", "ref_r")}
    t0 = time.time()
    with fixed_rng(a.seed):
        for j, (x, y, idx) in enumerate(dl):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            out = generate(G, x, mode, amp)
            r["l1"].append(float((out - y).abs().mean()))
            if not (ssim or f1 or fid_dir):
                continue
            o, t, sk = to_rgb8(out[0]), to_rgb8(y[0]), to_gray8(x[0])
            if ssim:
                r["ssim"].append(ssim_rgb(o, t))
            if f1:
                for key, img in (("", o), ("ref_", t)):
                    f, p_, rc = edge_f1(sk, img, a.canny_low, a.canny_high, a.f1_tol)
                    r[key + "f1"].append(f); r[key + "p"].append(p_); r[key + "r"].append(rc)
            if fid_dir:
                name = f"{int(idx[0]):06d}.png"
                cv2.imwrite(os.path.join(fid_dir, "fake", name), o[..., ::-1])
                cv2.imwrite(os.path.join(fid_dir, "real", name), t[..., ::-1])
            if (j + 1) % 500 == 0:
                log(f"  {tag} {j + 1}/{len(paths)}")
    r["sec"] = time.time() - t0
    return r


def compute_fid(d, a):
    try:
        from cleanfid import fid
        return float(fid.compute_fid(os.path.join(d, "fake"), os.path.join(d, "real"), mode="clean",
                                     num_workers=a.num_workers, device=torch.device("cuda")))
    except Exception as e:  # missing package or tiny sets (smoke)
        log(f"WARNING: FID skipped: {e}")
        return None


@torch.no_grad()
def sample_grid(G, base, paths, a, mode, amp, path):
    """One row per image: sketch | target | output (256 px)."""
    from torchvision.utils import save_image
    ds, tiles = PairDataset(base, paths, a, False), []
    with fixed_rng(a.seed):
        for i in range(len(ds)):
            x, y, _ = ds[i]
            x, y = x[None].cuda(), y[None].cuda()
            out = generate(G, x, mode, amp)
            for t in (x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x, y, out):
                tiles.append(F.interpolate(t.float(), size=256, mode="area"))
    save_image((torch.cat(tiles) + 1) / 2, path, nrow=3, padding=4, pad_value=1.0)


# ----------------------------------------------------------------------------- setup and checkpoints
def build(a, amp):
    G = UNetGenerator(a.sketch_channels, 3, a.ngf, bool(a.bottleneck_bn)).cuda()
    D = PatchGAN70(a.sketch_channels + 3, a.ndf).cuda()
    G.apply(init_weights)
    D.apply(init_weights)
    vgg = VGGPerceptual(a.vgg_layer).cuda() if a.use_perc and a.lambda_perc > 0 else None
    betas = (a.beta1, a.beta2)
    return {"G": G, "D": D, "vgg": vgg, "amp": amp,
            "optG": torch.optim.Adam(G.parameters(), lr=a.lr, betas=betas),
            "optD": torch.optim.Adam(D.parameters(), lr=a.lr, betas=betas),
            "scG": grad_scaler(amp), "scD": grad_scaler(amp)}


CRITICAL = ["size", "sketch_channels", "ngf", "ndf", "bottleneck_bn", "batch", "lambda_l1", "lambda_perc",
            "vgg_layer", "use_gan", "use_l1", "use_perc", "lr", "seed", "flip_p", "thicken_p", "select"]


def save_ckpt(path, m, state, a):
    torch.save({k: m[k].state_dict() for k in ("G", "D", "optG", "optD", "scG", "scD")} |
               {"amp": m["amp"], "state": state, "config": vars(a)}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_ckpt(path, m, a):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    diff = {k: (ck["config"].get(k), getattr(a, k)) for k in CRITICAL if ck["config"].get(k) != getattr(a, k)}
    if diff:
        log(f"WARNING: config differs from checkpoint (old, new): {diff}")
    for k in ("G", "D", "optG", "optD"):
        m[k].load_state_dict(ck[k])
    if ck["amp"] != m["amp"]:
        m["amp"] = ck["amp"]
        m["scG"], m["scD"] = grad_scaler(m["amp"]), grad_scaler(m["amp"])
    else:
        m["scG"].load_state_dict(ck["scG"])
        m["scD"].load_state_dict(ck["scD"])
    return ck["state"]


def save_generator(G, a, out, state, mode_info=None):
    """generator.pth = G weights + what the web app needs."""
    meta = {"arch": "UNetGenerator (8 levels)", "sketch_channels": a.sketch_channels, "ngf": a.ngf,
            "bottleneck_bn": a.bottleneck_bn, "size": a.size, "test_mode": a.test_mode,
            "test_mode_selection": mode_info, "best_epoch": state.get("best_epoch") if state else None,
            "select": f"val_{a.select}"}
    torch.save({"state_dict": G.state_dict(), **meta}, os.path.join(out, "generator.pth"))
    write_json(os.path.join(out, "generator_config.json"), meta)


def load_generator(a, out):
    path = a.ckpt or os.path.join(out, "checkpoints", "best.pt")
    if not os.path.exists(path):
        path = os.path.join(out, "generator.pth")
    ck = torch.load(path, map_location="cuda", weights_only=False)
    if "state_dict" in ck:
        sd, cfg, ep = ck["state_dict"], ck, ck.get("best_epoch")
        if a.test_mode == "auto" and ck.get("test_mode") in ("paper", "eval"):
            a.test_mode = ck["test_mode"]
    else:
        sd, cfg, ep = ck["G"], ck.get("config", {}), ck.get("epoch")
    for k in ("sketch_channels", "ngf", "bottleneck_bn", "size"):
        if k in cfg:
            setattr(a, k, cfg[k])
    G = UNetGenerator(a.sketch_channels, 3, a.ngf, bool(a.bottleneck_bn)).cuda()
    G.load_state_dict(sd)
    log(f"Loaded G: {path} (epoch {ep})")
    return G, path, ep


def pick_test_mode(G, base, val, a, amp):
    """auto: try both test-time modes on validation, keep the lower L1 (test set not used)."""
    res = {}
    for mode in ("eval", "paper"):
        r = evaluate_set(G, base, val, a, mode, amp, ssim=True, f1=False, tag=f"val-{mode}")
        res[mode] = {"val_l1": round(float(np.mean(r["l1"])), 5), "val_ssim": round(float(np.mean(r["ssim"])), 5)}
    best = min(res, key=lambda k: res[k]["val_l1"])
    log(f"TEST MODE (val, n={len(val)}): {res} -> {best}")
    return best, {"method": "lower validation L1", "val_n": len(val), "results": res, "selected": best}


# ----------------------------------------------------------------------------- training step
LOSSES = ["G_GAN", "G_L1", "G_perc", "D_real", "D_fake", "G_total"]


def train_step(m, x, y, a, bce):
    """One D step, then one G step. Returns losses or None on NaN/Inf."""
    G, D, amp = m["G"], m["D"], m["amp"]
    k = dict.fromkeys(LOSSES, 0.0)
    with autocast(amp):
        fake = G(x)
    if a.use_gan:  # D: [BCE(D(x,y),1) + BCE(D(x,G(x)),0)] / 2
        D.requires_grad_(True)
        with autocast(amp):
            pr, pf = D(x, y), D(x, fake.detach())
            l_real, l_fake = bce(pr, torch.ones_like(pr)), bce(pf, torch.zeros_like(pf))
            lD = 0.5 * (l_real + l_fake)
        if not torch.isfinite(lD):
            return None
        m["optD"].zero_grad(set_to_none=True)
        m["scD"].scale(lD).backward()
        m["scD"].step(m["optD"])
        m["scD"].update()
        k["D_real"], k["D_fake"] = l_real.item(), l_fake.item()
        D.requires_grad_(False)
    with autocast(amp):  # G: non-saturating GAN + lambda1 * L1 + lambda2 * perceptual
        lG = torch.zeros((), device=x.device)
        if a.use_gan:
            pg = D(x, fake)
            g = bce(pg, torch.ones_like(pg))
            lG, k["G_GAN"] = lG + g, g.item()
        if a.use_l1:
            l1 = F.l1_loss(fake.float(), y)
            lG, k["G_L1"] = lG + a.lambda_l1 * l1, l1.item()
        if m["vgg"] is not None:
            lp = m["vgg"](fake, y)
            lG, k["G_perc"] = lG + a.lambda_perc * lp, lp.item()
    if not torch.isfinite(lG):
        return None
    m["optG"].zero_grad(set_to_none=True)
    m["scG"].scale(lG).backward()
    m["scG"].step(m["optG"])
    m["scG"].update()
    k["G_total"] = lG.item()
    return k


def on_nan(m, state):
    if not m["amp"]:
        raise RuntimeError("NaN/Inf loss with AMP off; stopping.")
    log("WARNING: NaN/Inf loss -> AMP disabled permanently, batch skipped.")
    m["amp"] = False
    m["scG"], m["scD"] = grad_scaler(False), grad_scaler(False)
    state["nan"] = state.get("nan", 0) + 1


# ----------------------------------------------------------------------------- training loop
def train(a, out, split, target_epoch, subset=None):
    base = split["root_now"]
    tr = (subset or split)["train"]
    va = (subset or split)["val"]
    va_sel = va[: a.val_n] if a.val_n > 0 else va
    ck_dir, smp_dir = os.path.join(out, "checkpoints"), os.path.join(out, "samples")
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(smp_dir, exist_ok=True)

    seed_all(a.seed)
    m = build(a, bool(a.amp))
    state = {"epoch": 0, "best_epoch": None, "best_value": None, "train_sec": 0.0, "nan": 0, "sessions": 0}
    last = os.path.join(ck_dir, "last.pt")
    rpath = last if a.resume == "auto" and os.path.exists(last) else (
        a.resume if a.resume not in ("auto", "none") else None)
    if rpath:
        state = load_ckpt(rpath, m, a)
        log(f"RESUME: {rpath} -> {state['epoch']} epochs done")
        if os.path.abspath(rpath) != os.path.abspath(last):  # resuming from another session's output
            old_ck = os.path.dirname(rpath)
            old_root = os.path.dirname(old_ck)
            for src, dst in [(os.path.join(old_ck, "best.pt"), os.path.join(ck_dir, "best.pt")),
                             (os.path.join(old_root, "losses.csv"), os.path.join(out, "losses.csv")),
                             (os.path.join(old_root, "val_log.csv"), os.path.join(out, "val_log.csv"))]:
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy(src, dst)
            if os.path.isdir(os.path.join(old_root, "samples")):
                shutil.copytree(os.path.join(old_root, "samples"), smp_dir, dirs_exist_ok=True)
    state["sessions"] = state.get("sessions", 0) + 1
    if state["epoch"] >= target_epoch:
        return state, True

    ds = PairDataset(base, tr, a, True)
    gen = torch.Generator()
    dl = make_loader(ds, a, a.batch, True, gen=gen, drop_last=a.batch > 1)
    bce = nn.BCEWithLogitsLoss()
    loss_csv, val_csv = os.path.join(out, "losses.csv"), os.path.join(out, "val_log.csv")
    if not os.path.exists(loss_csv):
        with open(loss_csv, "w", newline="") as f:
            csv.writer(f).writerow(["type", "epoch", "step", "global_step"] + LOSSES + ["amp", "sec_per_step"])
    if not os.path.exists(val_csv):
        with open(val_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "val_l1", "val_ssim", "val_n", "mode", "epoch_sec", "selected"])

    t_session, last_ep_sec = time.time(), None
    log(f"TRAIN: {len(ds)} pairs, batch {a.batch}, {len(dl)} steps/epoch, epochs {state['epoch'] + 1}->{target_epoch}")
    for ep in range(state["epoch"] + 1, target_epoch + 1):
        if last_ep_sec and (time.time() - t_session + last_ep_sec) / 3600 > a.max_hours:
            log(f"TIME LIMIT ({a.max_hours} h): stopping before epoch {ep}; resume with --resume <out>/checkpoints/last.pt")
            return state, False
        gen.manual_seed(a.seed + ep)  # same shuffle order after resume
        m["G"].train(); m["D"].train()
        t_ep = time.time()
        win, n_win, t_win, tot, n_tot = {}, 0, time.time(), {}, 0
        for step, (x, y, _) in enumerate(dl, 1):
            k = train_step(m, x.cuda(non_blocking=True), y.cuda(non_blocking=True), a, bce)
            if k is None:
                on_nan(m, state)
                continue
            for kk, v in k.items():
                win[kk] = win.get(kk, 0) + v
                tot[kk] = tot.get(kk, 0) + v
            n_win += 1
            n_tot += 1
            if step % a.log_every == 0 or step == len(dl):
                avg = {kk: v / n_win for kk, v in win.items()}
                sps = (time.time() - t_win) / n_win
                with open(loss_csv, "a", newline="") as f:
                    csv.writer(f).writerow(["step", ep, step, (ep - 1) * len(dl) + step] +
                                           [round(avg.get(c, 0), 5) for c in LOSSES] + [int(m["amp"]), round(sps, 4)])
                if step % (a.log_every * 5) == 0 or step == len(dl):
                    log(f"ep {ep} {step}/{len(dl)} " + " ".join(f"{c}={avg.get(c, 0):.4f}" for c in LOSSES[:5])
                        + f" | {sps:.3f} s/step, ~{(len(dl) - step) * sps / 60:.1f} min left")
                win, n_win, t_win = {}, 0, time.time()
        with open(loss_csv, "a", newline="") as f:
            csv.writer(f).writerow(["epoch", ep, len(dl), ep * len(dl)] +
                                   [round(tot.get(c, 0) / max(n_tot, 1), 5) for c in LOSSES] +
                                   [int(m["amp"]), round((time.time() - t_ep) / max(n_tot, 1), 4)])

        # validation and model selection (test set never used here)
        selected, vl, vs = False, None, None
        if ep % a.val_every == 0 or ep == target_epoch:
            r = evaluate_set(m["G"], base, va_sel, a, active_mode(a), m["amp"], ssim=True, f1=False, tag="val")
            vl, vs = float(np.mean(r["l1"])), float(np.mean(r["ssim"]))
            v = vl if a.select == "l1" else vs
            if state["best_value"] is None or (v < state["best_value"] if a.select == "l1" else v > state["best_value"]):
                state["best_value"], state["best_epoch"], selected = v, ep, True
                torch.save({"G": m["G"].state_dict(), "epoch": ep, "val_l1": vl, "val_ssim": vs, "config": vars(a)},
                           os.path.join(ck_dir, "best.pt"))
            log(f"VAL ep {ep}: L1={vl:.4f} SSIM={vs:.4f} (n={len(va_sel)}){' <- BEST' if selected else ''}")

        ep_sec = time.time() - t_ep
        last_ep_sec = ep_sec
        state["epoch"] = ep
        state["train_sec"] += ep_sec
        with open(val_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, vl, vs, len(va_sel), active_mode(a), round(ep_sec, 1), int(selected)])
        if ep % a.sample_every == 0 or ep == target_epoch:
            sample_grid(m["G"], base, va[:8], a, active_mode(a), m["amp"], os.path.join(smp_dir, f"epoch_{ep:03d}.png"))
        if ep % a.ckpt_every == 0 or ep == target_epoch:
            save_ckpt(last, m, state, a)
        if ep % a.keep_every == 0:
            save_ckpt(os.path.join(ck_dir, f"epoch_{ep:03d}.pt"), m, state, a)
            for old in sorted(f for f in os.listdir(ck_dir) if f.startswith("epoch_"))[: -a.keep_last]:
                os.remove(os.path.join(ck_dir, old))
        log(f"EPOCH {ep} done: {ep_sec / 60:.1f} min, total {state['train_sec'] / 3600:.2f} h")
    return state, True


# ----------------------------------------------------------------------------- test evaluation
def test_eval(a, out, split, G, ck_path, ep, state=None, subset=None, mode_info=None):
    base = split["root_now"]
    te = (subset or split)["test"]
    mode, amp = active_mode(a), bool(a.amp)
    train_cfg = {}
    last = os.path.join(out, "checkpoints", "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        train_cfg, state = ck["config"], state or ck["state"]
        del ck
    fid_dir = tempfile.mkdtemp(prefix="fid_") if a.fid else None
    if fid_dir:
        os.makedirs(os.path.join(fid_dir, "fake"))
        os.makedirs(os.path.join(fid_dir, "real"))
    log(f"TEST: {len(te)} images, test_mode={mode}")
    r = evaluate_set(G, base, te, a, mode, amp, ssim=True, f1=True, fid_dir=fid_dir, tag="test")
    fid_val = compute_fid(fid_dir, a) if fid_dir else None
    n_fid = len(os.listdir(os.path.join(fid_dir, "fake"))) if fid_dir else 0
    if fid_dir:
        shutil.rmtree(fid_dir, ignore_errors=True)
    os.makedirs(os.path.join(out, "samples"), exist_ok=True)
    sample_grid(G, base, te[:8], a, mode, amp, os.path.join(out, "samples", "test_samples.png"))

    met = {
        "model": "pix2pix (8-level U-Net + 70x70 PatchGAN)", "mode": a.mode,
        "date": datetime.now().isoformat(timespec="seconds"), "split_evaluated": "test",
        "checkpoint": ck_path, "checkpoint_epoch": ep, "test_mode": mode, "test_mode_selection": mode_info,
        "counts": split["counts"] if not subset else {k: len(v) for k, v in subset.items()},
        "ssim": {**summary(r["ssim"]), "range": "[0, 1]", "color": "RGB", "lib": "scikit-image"},
        "fid": {"value": None if fid_val is None else round(fid_val, 4), "lib": "clean-fid", "mode": "clean",
                "n_fake": n_fid, "n_real": n_fid},
        "edge_f1": {**summary(r["f1"]), "precision": round(float(np.mean(r["p"])), 4),
                    "recall": round(float(np.mean(r["r"])), 4), "canny": [a.canny_low, a.canny_high],
                    "tolerance_px": a.f1_tol},
        "edge_f1_ground_truth": {**summary(r["ref_f1"]), "precision": round(float(np.mean(r["ref_p"])), 4),
                                 "recall": round(float(np.mean(r["ref_r"])), 4)},
        "l1": summary(r["l1"]),
        "training": None if state is None else {
            "batch": train_cfg.get("batch", a.batch), "target_epochs": train_cfg.get("epochs", a.epochs),
            "epochs_done": state["epoch"], "best_epoch": state["best_epoch"],
            "select": f"val_{train_cfg.get('select', a.select)}", "best_value": state["best_value"],
            "train_hours": round(state["train_sec"] / 3600, 3), "sessions": state.get("sessions"),
            "nan_events": state.get("nan", 0), "stopped_by_time_limit": state.get("time_limited", False)},
        "env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
        "config": vars(a),
    }
    write_json(os.path.join(out, "metrics.json"), met)
    log(f"SSIM {met['ssim']['mean']} +- {met['ssim']['std']} | FID {met['fid']['value']} (n={n_fid}) | "
        f"edge F1 {met['edge_f1']['mean']} +- {met['edge_f1']['std']}")
    return met


def finish(a, out, split, state, subset=None):
    """Load best G, pick test mode on validation (auto), write generator.pth, evaluate on test."""
    va = (subset or split)["val"]
    va = va[: a.val_n] if a.val_n > 0 else va
    G, ck_path, ep = load_generator(a, out)
    info = None
    if a.test_mode == "auto":
        a.test_mode, info = pick_test_mode(G, split["root_now"], va, a, bool(a.amp))
    save_generator(G, a, out, state, info)
    return test_eval(a, out, split, G, ck_path, ep, state, subset, info)


# ----------------------------------------------------------------------------- benchmark
def benchmark(a, out, split):
    base = split["root_now"]
    n_tr, n_va, n_te = (split["counts"][k] for k in ("train", "val", "test"))
    val_n = min(a.val_n, n_va) if a.val_n > 0 else n_va
    bce, res = nn.BCEWithLogitsLoss(), []
    for b in [int(v) for v in a.bench_batch.split(",")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seed_all(a.seed)
        a.batch = b
        try:
            m = build(a, bool(a.amp))
            need = (a.bench_warmup + a.bench_steps + 1) * b
            dl = make_loader(PairDataset(base, split["train"][:need], a, True), a, b, True, drop_last=True)
            t0, n = None, 0
            for i, (x, y, _) in enumerate(dl):
                if i == a.bench_warmup:
                    torch.cuda.synchronize(); t0 = time.time()
                if train_step(m, x.cuda(), y.cuda(), a, bce) is None:
                    on_nan(m, {})
                n += t0 is not None
                if n >= a.bench_steps:
                    break
            torch.cuda.synchronize()
            sps = (time.time() - t0) / n
            mem = torch.cuda.max_memory_allocated() / 2 ** 30
            r = evaluate_set(m["G"], base, split["val"][:50], a, "eval", m["amp"])
            spi = r["sec"] / 50
            steps = n_tr // b if b > 1 else n_tr
            ep_sec = sps * steps + spi * val_n / a.val_every
            hours = ep_sec * a.epochs / 3600
            s = {"batch": b, "sec_per_step": round(sps, 4), "steps_per_epoch": steps,
                 "epoch_min": round(ep_sec / 60, 1), "epochs_in_max_hours": int(a.max_hours * 3600 // ep_sec),
                 "target_epochs": a.epochs, "target_hours": round(hours, 2),
                 "test_min": round(spi * n_te / 60 + 5, 1), "sessions": max(1, math.ceil(hours / a.max_hours)),
                 "peak_mem_gb": round(mem, 2)}
            del m
        except torch.cuda.OutOfMemoryError:
            s = {"batch": b, "error": "OOM"}
            torch.cuda.empty_cache()
        res.append(s)
        log(f"BENCHMARK {s}")
    write_json(os.path.join(out, "benchmark.json"), {"results": res, "n_train": n_tr, "config": vars(a)})


# ----------------------------------------------------------------------------- main
def main():
    a = parse_args()
    if not os.path.exists("/kaggle") and not a.force_local:
        sys.exit("Intended for Kaggle; pass --force_local 1 to run elsewhere.")
    if not torch.cuda.is_available():
        sys.exit("ERROR: no CUDA GPU.")
    torch.backends.cudnn.benchmark = True
    a.out = a.out or ("/kaggle/working/smoke" if a.mode == "smoke" else "/kaggle/working")
    if a.mode == "smoke" and os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out, exist_ok=True)
    log(f"MODE={a.mode} out={a.out} GPU={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    base, files, tr_dir, va_dir = find_data(a.data_root)
    log(f"Data: {tr_dir} + {va_dir} -> {len(files)} pairs")
    check_halves(base, files, a.out, a.seed, a.color_side)
    split = make_split(files, a.split, a.seed, a.out, base)
    split["root_now"] = base
    log(f"SPLIT: {split['counts']} (seed {a.seed})")

    if a.mode == "benchmark":
        benchmark(a, a.out, split)
    elif a.mode == "smoke":
        n = a.smoke_n
        sub = {"train": split["train"][:n], "val": split["val"][: max(n // 4, 8)], "test": split["test"][: max(n // 4, 8)]}
        a.val_n, a.sample_every, a.ckpt_every, a.keep_every, a.log_every = 0, 1, 1, 1, 2
        train(a, a.out, split, 1, sub)
        a.resume = "auto"
        state, _ = train(a, a.out, split, 2, sub)  # rebuilt from last.pt
        assert state["epoch"] == 2, "resume failed"
        finish(a, a.out, split, state, sub)
        need = ["split.json", "losses.csv", "val_log.csv", "metrics.json", "generator.pth",
                "checkpoints/last.pt", "checkpoints/best.pt", "samples/epoch_002.png"]
        missing = [f for f in need if not os.path.exists(os.path.join(a.out, f))]
        if missing:
            sys.exit(f"SMOKE TEST FAILED: missing {missing}")
        log("SMOKE TEST PASSED")
    elif a.mode == "train":
        state, done = train(a, a.out, split, a.epochs)
        state["time_limited"] = not done
        finish(a, a.out, split, state)
    elif a.mode == "evaluate":
        G, ck_path, ep = load_generator(a, a.out)
        info = None
        if a.test_mode == "auto":
            va = split["val"][: a.val_n] if a.val_n > 0 else split["val"]
            a.test_mode, info = pick_test_mode(G, split["root_now"], va, a, bool(a.amp))
        test_eval(a, a.out, split, G, ck_path, ep, mode_info=info)


if __name__ == "__main__":
    main()
