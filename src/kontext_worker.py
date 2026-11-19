# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "torch==2.9.*",
#   "diffusers",
#   "transformers",
#   "accelerate",
#   "safetensors",
#   "sentencepiece",
#   "protobuf",
#   "numpy",
#   "opencv-python-headless",
#   "nunchaku @ https://github.com/nunchaku-tech/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu12.8torch2.9-cp312-cp312-win_amd64.whl ; sys_platform == 'win32'",
#   "nunchaku @ https://github.com/nunchaku-tech/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu12.8torch2.9-cp312-cp312-linux_x86_64.whl ; sys_platform == 'linux'",
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
AI restyle worker: FLUX.1 Kontext [dev], an instruction-following image editor,
4-bit quantised with Nunchaku (SVDQuant) so it fits a 12-16 GB GPU.

Unlike the SD 1.5 worker this needs no control image: the prompt simply says
"turn this photo into a line drawing" and the model redraws it.

Weights (cached in ~/.cache/huggingface, ~10 GB on first start):
- 4-bit transformer and T5 text encoder from nunchaku-ai (not gated)
- CLIP text encoder, VAE and tokenizers from black-forest-labs/FLUX.1-Kontext-dev,
  which is GATED: accept the licence on its Hugging Face page, then log in once
  with  uvx hf auth login  (or set HF_TOKEN). Licence: FLUX.1 [dev] non-commercial.

Same JSON-lines protocol as diffusion_worker.py; request keys:
    "prompt", "steps", "guidance", "size" (longest side in px), "seed"

Self-test:  uv run src/kontext_worker.py --selftest images/monalisa.png
            (writes images/monalisa_kontext.png)
"""

import os
import sys

from diffusion_worker import fit_size, log, read_bgr, serve

# GPUs below this much VRAM get the low-memory mode (layer offload). Without it the whole
# pipeline (~10 GB) doesn't fit an 8 GB card, and the Windows driver silently spills into
# shared system RAM instead of failing: one image then takes ~50 min instead of minutes.
# Override with KONTEXT_LOW_VRAM=1 / 0.
LOW_VRAM_BELOW_GB = 12

KONTEXT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
NUNCHAKU_TRANSFORMER = "nunchaku-ai/nunchaku-flux.1-kontext-dev/svdq-{precision}_r32-flux.1-kontext-dev.safetensors"
NUNCHAKU_T5 = "nunchaku-ai/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"

DEFAULTS = {
    "prompt": ("Convert this photo into a clean black and white line drawing, colouring book style. "
               "Thin uniform black outlines on a plain white background. "
               "No shading, no hatching, no grey, no colour fill. Keep the faces and pose."),
    "steps": 20,
    "guidance": 2.5,
    "size": 1024,
    "seed": 42,
}


def load_pipeline():
    import torch
    from diffusers import FluxKontextPipeline
    from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
    from nunchaku.utils import get_precision

    if not torch.cuda.is_available():
        raise RuntimeError("Nunchaku needs an NVIDIA GPU with CUDA")
    precision = get_precision()  # 'int4' (Turing/Ampere/Ada) or 'fp4' (Blackwell)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    env = os.environ.get("KONTEXT_LOW_VRAM")
    low_vram = env == "1" if env in ("0", "1") else vram_gb < LOW_VRAM_BELOW_GB
    log(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB), "
        f"nunchaku {precision}, {'low-memory mode (layer offload)' if low_vram else 'all on GPU'}")

    log("Loading 4-bit FLUX.1 Kontext transformer ...")
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        NUNCHAKU_TRANSFORMER.format(precision=precision), offload=low_vram)
    extra = {}
    if not low_vram:
        # 4-bit T5 lives on the GPU. In low-memory mode the standard T5 from the Kontext repo
        # is used instead and streamed layer by layer (as in Nunchaku's offload examples).
        log("Loading 4-bit T5 text encoder ...")
        extra["text_encoder_2"] = NunchakuT5EncoderModel.from_pretrained(NUNCHAKU_T5)
    log(f"Loading CLIP / VAE{'' if extra else ' / T5'} from {KONTEXT_MODEL} ...")
    try:
        # components passed in are not downloaded, so this fetches only the missing ones
        pipe = FluxKontextPipeline.from_pretrained(
            KONTEXT_MODEL, transformer=transformer, torch_dtype=torch.bfloat16,
            use_safetensors=True, **extra)
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
            raise RuntimeError(
                f"{KONTEXT_MODEL} is gated: accept its licence at "
                f"https://huggingface.co/{KONTEXT_MODEL} and run 'uvx hf auth login'") from e
        raise
    pipe.set_progress_bar_config(disable=True)
    if low_vram:
        pipe.enable_sequential_cpu_offload()   # each layer visits the GPU only while it runs
    else:
        pipe = pipe.to("cuda")
    return pipe, "cuda"


def log_steps(pipe, step, timestep, callback_kwargs):
    """Per-step progress on stderr; the GUI shows the latest log line while it waits."""
    log(f"step {step + 1}/{pipe.num_timesteps}")
    return callback_kwargs


def generate(pipe, device, req):
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    p = {**DEFAULTS, **{k: v for k, v in req.items() if v is not None}}
    src = fit_size(read_bgr(req["in"]), int(p["size"]), multiple=16)  # FLUX needs multiples of 16
    h, w = src.shape[:2]
    image = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    out = pipe(
        image=image, prompt=p["prompt"], height=h, width=w,
        num_inference_steps=int(p["steps"]), guidance_scale=float(p["guidance"]),
        generator=torch.Generator(device=device).manual_seed(int(p["seed"])),
        # Output size = input size, so the drawing lines up pixel for pixel with the photo
        # (line colours are taken from the photo). Without max_area the pipeline rescales
        # the output to ~1 MP while the input stays at h x w, which offsets the drawing.
        max_area=h * w,
        _auto_resize=False,  # keep our size and aspect instead of a preferred Kontext size
        callback_on_step_end=log_steps,
    ).images[0]
    bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    buf.tofile(req["out"])


if __name__ == "__main__":
    argv = sys.argv if len(sys.argv) < 3 else [*sys.argv[:3], "_kontext.png"]
    sys.exit(serve(load_pipeline, generate, argv))
