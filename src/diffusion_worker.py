# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch",
#   "diffusers",
#   "transformers",
#   "accelerate",
#   "safetensors",
#   "numpy",
#   "opencv-python-headless",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""
AI restyle worker: Stable Diffusion 1.5 image-to-image guided by a lineart ControlNet.

Runs as a separate, long-lived process so the GUI stays light and the model is
loaded only once. The first start downloads ~3 GB of model weights to the
Hugging Face cache (~/.cache/huggingface).

Protocol (one JSON object per line):
    stdout on start:  {"ready": true, "device": "...", "load_s": ...}  or {"ready": false, "error": "..."}
    stdin request:    {"in": "<png path>", "out": "<png path>", "prompt": "...", "negative": "...",
                       "strength": 0.55, "steps": 20, "guidance": 7.0, "control_scale": 1.0, "seed": 42}
    stdout reply:     {"ok": true, "out": "<png path>", "seconds": 2.1}  or {"ok": false, "error": "..."}
Log output goes to stderr.

Self-test:  uv run src/diffusion_worker.py --selftest images/monalisa.png
kontext_worker.py speaks the same protocol and reuses serve() and the helpers here.
"""

import json
import os
import sys
import time
import traceback

# Windows without Developer Mode can't symlink in the HF cache; that works, just noisily
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import cv2
import numpy as np

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_lineart"
SIZE = 512  # longest side of the generated image (SD 1.5 native resolution)

DEFAULTS = {
    "prompt": "comic book illustration, bold black ink outlines, flat colours, clean line art",
    "negative": "blurry, photo, noisy, grainy, text, watermark, signature",
    "strength": 0.55,
    "steps": 20,
    "guidance": 7.0,
    "control_scale": 1.0,
    "seed": 42,
}


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def read_bgr(path):
    data = np.fromfile(path, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read {path}")
    return img


def fit_size(img, size=SIZE, multiple=8):
    """Resize so the longest side is about `size` and both sides are multiples of `multiple`."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    m = multiple
    nw, nh = max(m, round(w * s / m) * m), max(m, round(h * s / m) * m)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def lineart_control(bgr):
    """Control image for the lineart ControlNet: white lines on black.
    Uses the same difference-of-Gaussians ink lines as the comic style."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 40, 9).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(gray, (0, 0), 1.6)
    lines = np.clip((g2 - g1) * 25.0, 0, 1)          # soft lines, like the lineart annotator
    return cv2.cvtColor((lines * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)


def load_pipeline():
    import torch
    from diffusers import (ControlNetModel, StableDiffusionControlNetImg2ImgPipeline,
                           UniPCMultistepScheduler)

    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    device = "cuda" if cuda else "cpu"
    log(f"torch {torch.__version__}, device {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if cuda else " - no CUDA, this will be very slow"))

    log(f"Loading {CONTROLNET_MODEL} ...")
    # use_safetensors: never fall back to pickle (.bin) weights, which can execute code on load
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=dtype,
                                                 use_safetensors=True)
    log(f"Loading {BASE_MODEL} ...")
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        BASE_MODEL, controlnet=controlnet, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False,
        variant="fp16" if cuda else None, use_safetensors=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    if cuda:
        pipe.enable_attention_slicing()  # keeps 8 GB laptop GPUs comfortable
    return pipe, device


def generate(pipe, device, req):
    import torch
    from PIL import Image

    p = {**DEFAULTS, **{k: v for k, v in req.items() if v is not None}}
    src = fit_size(read_bgr(req["in"]))
    init = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    control = Image.fromarray(lineart_control(src))
    generator = torch.Generator(device=device).manual_seed(int(p["seed"]))
    out = pipe(
        prompt=p["prompt"], negative_prompt=p["negative"],
        image=init, control_image=control,
        strength=float(p["strength"]), num_inference_steps=int(p["steps"]),
        guidance_scale=float(p["guidance"]),
        controlnet_conditioning_scale=float(p["control_scale"]),
        generator=generator,
    ).images[0]
    bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    buf.tofile(req["out"])


def serve(load, generate, argv):
    """Shared worker main loop (also used by kontext_worker.py).
    load() -> (pipe, device); generate(pipe, device, request_dict) writes request["out"]."""
    t0 = time.perf_counter()
    try:
        pipe, device = load()
    except Exception as e:
        traceback.print_exc()
        reply({"ready": False, "error": f"{type(e).__name__}: {e}"})
        return 1
    reply({"ready": True, "device": device, "load_s": round(time.perf_counter() - t0, 1)})

    if len(argv) >= 3 and argv[1] == "--selftest":
        src = argv[2]
        out = src.rsplit(".", 1)[0] + (argv[3] if len(argv) > 3 else "_ai.png")
        for i in range(2):  # second run shows the warm speed
            t = time.perf_counter()
            generate(pipe, device, {"in": src, "out": out})
            log(f"selftest run {i + 1}: {time.perf_counter() - t:.1f} s -> {out}")
        return 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        t = time.perf_counter()
        try:
            req = json.loads(line)
            generate(pipe, device, req)
            reply({"ok": True, "out": req["out"], "seconds": round(time.perf_counter() - t, 2)})
        except Exception as e:
            traceback.print_exc()
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
    return 0


def main():
    return serve(load_pipeline, generate, sys.argv)


if __name__ == "__main__":
    sys.exit(main())
