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

import sys

from diffusion_worker import fit_size, log, read_bgr, serve

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
    log(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, nunchaku {precision}")

    log("Loading 4-bit FLUX.1 Kontext transformer ...")
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        NUNCHAKU_TRANSFORMER.format(precision=precision))
    log("Loading 4-bit T5 text encoder ...")
    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(NUNCHAKU_T5)
    log(f"Loading CLIP / VAE from {KONTEXT_MODEL} ...")
    try:
        # components passed in are not downloaded, so this fetches only CLIP, VAE, tokenizers
        pipe = FluxKontextPipeline.from_pretrained(
            KONTEXT_MODEL, transformer=transformer, text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16, use_safetensors=True)
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
            raise RuntimeError(
                f"{KONTEXT_MODEL} is gated: accept its licence at "
                f"https://huggingface.co/{KONTEXT_MODEL} and run 'uvx hf auth login'") from e
        raise
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")
    return pipe, "cuda"


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
        _auto_resize=False,  # keep our size and aspect instead of snapping to ~1 MP
    ).images[0]
    bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    buf.tofile(req["out"])


if __name__ == "__main__":
    argv = sys.argv if len(sys.argv) < 3 else [*sys.argv[:3], "_kontext.png"]
    sys.exit(serve(load_pipeline, generate, argv))
