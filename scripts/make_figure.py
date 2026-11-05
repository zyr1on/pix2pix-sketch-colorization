"""Sample figure: one row per image -> original sketch | ground truth | pix2pix.
Inputs are dataset pairs (left = color, right = sketch). Only TEST-split images are used (results/split.json).

    python scripts/make_figure.py --input <folder_with_pairs>
"""
import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
from model import Colorizer, read_rgb  # noqa: E402

TITLES = ["Original sketch", "Ground truth", "pix2pix"]


def split_of(split_path):
    s = json.load(open(split_path, encoding="utf-8"))
    names = {}
    for part in ("train", "val", "test"):
        for rel in s[part]:
            names.setdefault(os.path.basename(rel), set()).add(part)
    return names


def draw(m, rows, path, dpi):
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows) + 0.4), squeeze=False)
    for i, p in enumerate(rows):
        img = read_rgb(p)
        if img.shape[1] != 2 * img.shape[0]:
            sys.exit(f"ERROR: {p} is not a pair (width must be 2 x height)")
        truth, sketch = m.split_pair(img)
        for j, im in enumerate((sketch, truth, m.colorize(sketch))):
            ax = axes[i, j]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(TITLES[j], fontsize=13)
    plt.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.4)
    plt.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="folder with dataset pairs")
    p.add_argument("--weights", default=os.path.join(ROOT, "weights", "generator.pth"))
    p.add_argument("--split", default=os.path.join(ROOT, "results", "split.json"))
    p.add_argument("--out", default=os.path.join(ROOT, "assets", "samples.png"))
    p.add_argument("--test_only", type=int, default=1)
    p.add_argument("--max", type=int, default=6, help="total rows")
    p.add_argument("--rows", type=int, default=3, help="rows per file (6 rows -> samples_a.png, samples_b.png)")
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()

    files = sorted(f for e in ("png", "jpg", "jpeg") for f in glob.glob(os.path.join(a.input, f"*.{e}")))
    names, chosen = split_of(a.split), []
    for f in files:
        parts = names.get(os.path.basename(f), set())
        if a.test_only and parts != {"test"}:
            print(f"skipped ({'/'.join(sorted(parts)) or 'not in split'}): {os.path.basename(f)}")
            continue
        chosen.append(f)
    chosen = chosen[: a.max]
    if not chosen:
        sys.exit("ERROR: no test-split images found.")

    m = Colorizer(a.weights)
    groups = [chosen[i:i + a.rows] for i in range(0, len(chosen), a.rows)]
    stem, ext = os.path.splitext(a.out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for g, rows in enumerate(groups):
        path = a.out if len(groups) == 1 else f"{stem}_{chr(97 + g)}{ext}"
        draw(m, rows, path, a.dpi)
        print(f"saved {path}: {', '.join(os.path.basename(r) for r in rows)}")


if __name__ == "__main__":
    main()
