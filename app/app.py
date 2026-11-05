"""Gradio web app for pix2pix sketch colorization (inference only).

    python app/app.py                              # weights/generator.pth, opens http://127.0.0.1:7860
    python app/app.py --model path/to/generator.pth --port 7861 --no_browser

Input: a plain line-art sketch, or a dataset pair (left = color, right = sketch).
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from model import Colorizer  # noqa: E402

parser = argparse.ArgumentParser(description="pix2pix sketch colorization web app")
parser.add_argument("--model", default=os.environ.get("PIX2PIX_WEIGHTS", os.path.join(ROOT, "weights", "generator.pth")),
                    help="path to generator.pth (default: weights/generator.pth)")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--no_browser", action="store_true", help="do not open a browser tab")
ARGS, _ = parser.parse_known_args()
WEIGHTS = ARGS.model
AUTO, SKETCH, PAIR = "Auto detect", "Sketch only", "Pair (left color, right sketch)"

# Gradio runs each request on a different thread and CUDA/cuDNN set up per-thread state on first use
# (~0.5 s). All inference therefore runs on one warmed-up worker thread.
GPU = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")


def _sync(m):
    if m.device.type == "cuda":
        torch.cuda.synchronize()


def _infer(m, sketch):
    _sync(m)
    t0 = time.perf_counter()
    out = m.colorize(sketch)
    _sync(m)
    return out, (time.perf_counter() - t0) * 1000


def load():
    if not os.path.exists(WEIGHTS):
        sys.exit(f"ERROR: weights not found: {WEIGHTS} (download generator.pth, see README, or pass --model)")
    t0 = time.perf_counter()
    m = GPU.submit(Colorizer, WEIGHTS).result()
    load_s = time.perf_counter() - t0
    print(f"[1/2] model loaded ({load_s:.2f} s) on {m.device}")
    t0 = time.perf_counter()
    blank = np.full((m.size, m.size, 3), 255, np.uint8)
    for _ in range(3):
        GPU.submit(_infer, m, blank).result()
    warm_s = time.perf_counter() - t0
    print(f"[2/2] warm-up done ({warm_s:.2f} s)")
    return m, load_s, warm_s


M, LOAD_S, WARM_S = load()
DEVICE = torch.cuda.get_device_name(0) if M.device.type == "cuda" else "CPU"


def is_pair(img):
    h, w = img.shape[:2]
    return w == 2 * h


def colorize(img, kind):
    if img is None:
        raise gr.Error("Please upload an image.")
    img = np.asarray(img)
    img = np.stack([img] * 3, -1) if img.ndim == 2 else img[..., :3]
    pair = is_pair(img) if kind == AUTO else kind == PAIR
    if pair and not is_pair(img):
        raise gr.Error("A pair must be twice as wide as it is tall (left color, right sketch).")
    truth, sketch = M.split_pair(img) if pair else (None, img)
    out, ms = GPU.submit(_infer, M, sketch).result()
    info = f"Input: {'pair' if pair else 'sketch'} · {sketch.shape[1]}x{sketch.shape[0]} px · {ms:.0f} ms"
    return out, sketch, gr.update(value=truth, visible=pair), out, info


CSS = """
.gradio-container {max-width: 1100px !important; width: 100% !important; margin: 0 auto !important;}
#title {text-align: center;}
"""
# always open in light mode
LIGHT_JS = """
() => {
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u.href); }
}
"""

with gr.Blocks(title="pix2pix sketch colorization", fill_width=True) as demo:
    gr.Markdown("## Sketch Colorization (pix2pix)", elem_id="title")
    with gr.Row(equal_height=True):
        with gr.Column():
            inp = gr.Image(label="Upload sketch", type="numpy", image_mode="RGB",
                           sources=["upload", "clipboard"], height=440)
            kind = gr.Radio([AUTO, SKETCH, PAIR], value=AUTO, label="Input type")
            run = gr.Button("Colorize", variant="primary")
        with gr.Column():
            out = gr.Image(label="Output", type="numpy", height=440, interactive=False)
    with gr.Accordion("Details", open=False):
        info = gr.Markdown(f"Best epoch {M.meta.get('best_epoch')} · {DEVICE} · test mode `{M.mode}` · "
                           f"load {LOAD_S:.1f} s, warm-up {WARM_S:.1f} s")
        with gr.Row():
            d_sketch = gr.Image(label="Sketch", type="numpy", height=260, interactive=False)
            d_truth = gr.Image(label="Ground truth", type="numpy", height=260, interactive=False, visible=False)
            d_out = gr.Image(label="pix2pix", type="numpy", height=260, interactive=False)
    run.click(colorize, inputs=[inp, kind], outputs=[out, d_sketch, d_truth, d_out, info])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=ARGS.port, inbrowser=not ARGS.no_browser,
                theme=gr.themes.Default(primary_hue="blue", neutral_hue="gray"), css=CSS, js=LIGHT_JS)
