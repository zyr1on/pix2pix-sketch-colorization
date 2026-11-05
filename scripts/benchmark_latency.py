"""Inference latency on the local GPU -> results/latency.json.
Latency does not depend on image content (fixed 512x512 input), so a few images are enough.

    python scripts/benchmark_latency.py --input <folder_with_images>
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
from model import Colorizer, read_rgb  # noqa: E402


def stats(v):
    v = np.asarray(v) * 1000
    return {"mean_ms": round(float(v.mean()), 2), "std_ms": round(float(v.std()), 2),
            "median_ms": round(float(np.median(v)), 2), "p95_ms": round(float(np.percentile(v, 95)), 2),
            "n": int(v.size)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="sketches or dataset pairs")
    p.add_argument("--weights", default=os.path.join(ROOT, "weights", "generator.pth"))
    p.add_argument("--out", default=os.path.join(ROOT, "results", "latency.json"))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=50, help="runs per image")
    p.add_argument("--fp16", type=int, default=1)
    a = p.parse_args()
    if not torch.cuda.is_available():
        sys.exit("ERROR: no CUDA GPU.")

    files = sorted(f for e in ("png", "jpg", "jpeg") for f in glob.glob(os.path.join(a.input, f"*.{e}")))
    if not files:
        sys.exit(f"ERROR: no images in {a.input}")
    t0 = time.perf_counter()
    m = Colorizer(a.weights, fp16=a.fp16)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    sketches = []
    for f in files:
        img = read_rgb(f)
        sketches.append(m.split_pair(img)[1] if img.shape[1] == 2 * img.shape[0] else img)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    m.colorize(sketches[0])
    torch.cuda.synchronize()
    first_s = time.perf_counter() - t0
    for i in range(a.warmup):
        m.colorize(sketches[i % len(sketches)])

    e2e, net = [], []
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(a.repeat):
        for s in sketches:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            m.colorize(s)
            torch.cuda.synchronize()
            e2e.append(time.perf_counter() - t0)
            x = m.preprocess(s).to(m.device)
            torch.cuda.synchronize()
            ev0.record()
            m.forward(x)
            ev1.record()
            torch.cuda.synchronize()
            net.append(ev0.elapsed_time(ev1) / 1000)

    res = {"date": datetime.now().isoformat(timespec="seconds"), "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "cuda": torch.version.cuda, "fp16": m.fp16, "batch": 1,
           "test_mode": m.mode, "resolution": f"{m.size}x{m.size}", "images": len(files),
           "repeat_per_image": a.repeat, "load_s": round(load_s, 3), "first_call_s": round(first_s, 3),
           "end_to_end": {**stats(e2e), "scope": "preprocess + host-to-device + forward + postprocess"},
           "forward_only": {**stats(net), "scope": "generator forward (CUDA events)"}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"end-to-end {res['end_to_end']['mean_ms']} +- {res['end_to_end']['std_ms']} ms | "
          f"forward {res['forward_only']['mean_ms']} ms | first call {first_s * 1000:.0f} ms -> {a.out}")


if __name__ == "__main__":
    main()
