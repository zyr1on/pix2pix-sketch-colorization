# pix2pix-sketch-colorization

Automatic colorization of anime line-art sketches with pix2pix (Isola et al., 2017). The model takes only the sketch as input (no color hints) and outputs a 512x512 RGB image. Includes the Kaggle training code and a local Gradio web app.

## Results

Test split, 1,777 pairs, best checkpoint (epoch 16). Source: [`results/metrics.json`](results/metrics.json), [`results/latency.json`](results/latency.json).

| Metric | Value |
|---|---|
| SSIM (RGB, [0, 1]) | 0.854 ± 0.059 |
| FID (clean-fid, 1,777 vs 1,777) | 29.23 |
| Edge F1 (Canny 100/200, ±2 px) | 0.962 ± 0.021 (ground truth: 0.956) |
| L1 ([-1, 1] scale) | 0.121 ± 0.058 |
| Inference, RTX 4050 Laptop, fp16, batch 1 | 8.05 ± 0.35 ms per image (forward only 7.20 ms) |

Edge F1 compares the Canny edges of the sketch with those of the output; the same measure applied to the ground-truth image is given for reference.

![samples a](assets/samples_a.png)
![samples b](assets/samples_b.png)

Sketches come from the test split and were not seen during training or model selection. With the sketch as the only input, colors are often plausible but differ from the ground truth.

## Training

| | |
|---|---|
| Data | [Anime Sketch Colorization Pair](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair), each image = color (left) + sketch (right) |
| Split | train/ and val/ merged (17,769 pairs), shuffled with seed 42, 80/10/10 -> 14,215 / 1,777 / 1,777 |
| Input / output | 1-channel sketch, 512x512, [-1, 1] -> 3-channel RGB, Tanh |
| Augmentation | joint horizontal flip (p = 0.5), sketch-only line thickening by erosion (p = 0.3, 2-3 px), no random crop |
| Generator | 8-level U-Net, skip connections, dropout 0.5 in the first 3 decoder layers, no BN at the bottleneck |
| Discriminator | 70x70 PatchGAN on sketch + image |
| Loss | cGAN (non-saturating, D loss halved) + 100 · L1 + 10 · VGG-19 `relu4_4` perceptual loss |
| Optimizer | Adam, lr 2e-4, betas (0.5, 0.999), N(0, 0.02) init, AMP fp16 |
| Batch / epochs | 4 / 25 (best epoch 16, lowest validation L1 on 500 images) |
| Hardware / time | Kaggle, Tesla T4, 8.75 h of training in one session |

At test time dropout stays on and BatchNorm uses the statistics of the current image, as in the paper. This was chosen over `model.eval()` by validation L1 (0.1175 vs 0.1226).

Per-epoch validation used `model.eval()`, whose BatchNorm running statistics fluctuate during GAN training, so validation L1 varied strongly between epochs while the training loss decreased steadily.

## Usage

```bash
pip install -r requirements.txt   # install PyTorch with CUDA separately
```

The trained weights (`generator.pth`, 218 MB) are not tracked in git; place the file at `weights/generator.pth` (or set `PIX2PIX_WEIGHTS`).

Web app (opens http://127.0.0.1:7860). Upload a plain sketch or a dataset pair; the output is shown on the right and the sketch / ground truth / output under "Details":

```bash
python app/app.py
```

Training on Kaggle: import `train/pix2pix_kaggle.ipynb` (GPU T4, Internet on, dataset attached) and run *Save & Run All*, or call the script directly:

```bash
python train/train.py --mode smoke      # end-to-end check on a small subset
python train/train.py --mode benchmark  # epoch time and memory
python train/train.py --mode train      # batch 4, 25 epochs, resumable
python train/train.py --mode evaluate   # test metrics from a checkpoint
```

`train/build_notebook.py` regenerates the notebook after editing `train.py`.

Scripts (dataset pairs as input):

```bash
python scripts/benchmark_latency.py --input <folder>
python scripts/make_figure.py --input <folder>   # test-split images only
```

## Layout

```
train/     train.py, pix2pix_kaggle.ipynb, build_notebook.py
app/       app.py (Gradio), model.py (inference wrapper)
scripts/   benchmark_latency.py, make_figure.py
results/   metrics.json, latency.json, split.json
assets/    sample figures
weights/   generator.pth (not tracked)
```

## Reference

P. Isola, J.-Y. Zhu, T. Zhou, A. A. Efros. Image-to-Image Translation with Conditional Adversarial Networks. CVPR 2017.
