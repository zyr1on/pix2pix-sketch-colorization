"""Builds pix2pix_kaggle.ipynb with train.py embedded. Run: python build_notebook.py"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "train.py"), encoding="utf-8") as f:
    code = f.read()


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def py(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(True)}


cells = [
    md("""
# pix2pix sketch colorization (Kaggle, Tesla T4)
Settings: Accelerator = GPU T4, Internet = On, Input = `ktaebum/anime-sketch-colorization-pair`.
Run with *Save Version -> Save & Run All*: smoke test -> benchmark -> training -> test metrics -> plots -> zip.
"""),
    py("""
SMOKE = True       # tiny end-to-end check; the run stops if it fails
BENCHMARK = True   # epoch time + memory (benchmark.json)
TRAIN = True       # full training + test evaluation
EPOCHS = 25
BATCH = 4
MAX_HOURS = 10.5   # Kaggle sessions end at 12 h
RESUME = "auto"    # or "/kaggle/input/<previous-output>/checkpoints/last.pt"
"""),
    py("""
import glob, json, os, shutil, subprocess, zipfile
os.chdir("/kaggle/working")

def run(cmd):
    print(">>>", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"failed ({p.returncode}): {cmd}")

run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run("pip install -q clean-fid")
run("find /kaggle/input -maxdepth 5 -type d | grep -E '/(train|val)$'")
"""),
    py("%%writefile /kaggle/working/train.py\n" + code),
    py("""
if SMOKE:
    run(f"python train.py --mode smoke --batch {BATCH}")
"""),
    py("""
if BENCHMARK:
    run(f"python train.py --mode benchmark --bench_batch {BATCH} --epochs {EPOCHS} --max_hours {MAX_HOURS}")
"""),
    py("""
if TRAIN:
    run(f"python train.py --mode train --epochs {EPOCHS} --batch {BATCH} --max_hours {MAX_HOURS} --resume {RESUME}")
"""),
    py("""
# loss and validation curves
if TRAIN:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    k = pd.read_csv("losses.csv"); k = k[k.type == "step"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(k.global_step, k.G_L1); ax[0].set_title("G_L1")
    ax[1].plot(k.global_step, k.G_perc); ax[1].set_title("G_perc (VGG-19)")
    for c in ("G_GAN", "D_real", "D_fake"):
        ax[2].plot(k.global_step, k[c], label=c)
    ax[2].legend(); ax[2].set_title("Adversarial")
    for x in ax: x.set_xlabel("step")
    plt.tight_layout(); plt.savefig("samples/loss_curves.png", dpi=150); plt.close()
    v = pd.read_csv("val_log.csv").dropna(subset=["val_l1"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(v.epoch, v.val_l1, marker="o"); ax[0].set_ylabel("val L1")
    ax[1].plot(v.epoch, v.val_ssim, marker="o"); ax[1].set_ylabel("val SSIM")
    for x in ax: x.set_xlabel("epoch"); x.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("samples/val_curves.png", dpi=150); plt.close()
"""),
    py("""
# collect small outputs into results.zip (generator.pth is downloaded separately)
if TRAIN:
    with zipfile.ZipFile("results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["metrics.json", "split.json", "half_check.json", "benchmark.json", "losses.csv",
                  "val_log.csv", "generator_config.json", "smoke/metrics.json"]:
            if os.path.exists(f):
                z.write(f)
        for f in glob.glob("samples/*.png"):
            z.write(f)
    shutil.rmtree("smoke", ignore_errors=True)
    print(json.dumps({k: json.load(open("metrics.json"))[k] for k in ("ssim", "fid", "edge_f1", "counts")}, indent=2))
"""),
]

nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
    "nbformat": 4, "nbformat_minor": 4}
with open(os.path.join(HERE, "pix2pix_kaggle.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote pix2pix_kaggle.ipynb")
