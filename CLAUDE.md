# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

RobotPainter drives a Universal Robots (UR) arm to draw images. `src/image_to_urscript.py` converts an image into a line drawing and generates URScript programs from it. `urscript/URScript.script` is the original hand-written test program (a 100 mm square). There is no test suite.

Layout:
- `src/`: all code. The scripts sit side by side, because the GUI imports `image_to_urscript` and finds the AI worker scripts next to itself, and `kontext_worker.py` imports helpers from `diffusion_worker.py`.
- `images/`: input images.
- `examples/`: committed sample outputs. Regenerate them with the CLI after pipeline changes.
- `output/`: the GUI's save folder, gitignored.

## Commands

Python is not on PATH; use `uv`. Each script declares its dependencies inline (PEP 723), so `uv run` installs them automatically. Run from the repo root:

```
uv run src/image_to_urscript.py images/monalisa.png -o examples/edges/monalisa.script
uv run src/image_to_urscript.py images/monalisa.png --style comic -o examples/comic/monalisa.script
uv run src/image_to_urscript.py images/monalisa.png --style comic --colours black,green,red,blue -o examples/colour/monalisa.script
uv run src/image_to_urscript.py images/monalisa_kontext.png --style lineart -o examples/lineart/monalisa.script
```

GUI: `uv run src/robot_painter_gui.py`. It offers drag & drop (tkinterdnd2), live settings, webcam mode and a save button.

Windows Application Control on this machine blocks the `opencv-contrib-python` DLL and numpy's `numpy.random` extension (`_pcg64`). Stick to `opencv-python-headless` and implement anything from `cv2.ximgproc` and similar modules in numpy (e.g. `thin()` is a hand-written Zhang-Suen thinning).

Always check `<name>_preview.png` after a change. It is the only way to verify the output locally. The `.script` files run on the UR controller (Script node in PolyScope); no simulator is set up here.

## Image → URScript pipeline

Tunable properties are module-level constants at the top of `image_to_urscript.py`, grouped by style. Both styles produce a list of pixel-space polylines `(row, col)` that feed a shared back end.

- **edges**: blur → Canny → `trace_strokes`. This is a custom 8-neighbour walker over a 1-px line map, which avoids the doubled outlines `cv2.findContours` gives.
- **comic**: `equalizeHist` → two smoothing paths. Heavy bilateral smoothing feeds k-means posterising into `TONE_LEVELS` (index 0 = darkest). Light smoothing feeds the XDoG edge term (DoG threshold) for ink → `thin` → `trace_strokes`. Shading comes from `fill_strokes`, which works per connected region of (tone ≤ max_tone, pen) for each `HATCH_LAYERS` entry `(angle, spacing_mm, max_tone)`:
  - `scanlines` cuts the region into runs along the fill direction.
  - `chain_runs` groups runs of consecutive scanlines that overlap along the fill direction into chains, one strip of the region each.
  - `chained_strokes` turns each chain into pen paths, depending on `FILL_STYLE`:
    - `"zigzag"` (default): V-shaped diagonals through alternating run ends (start of even runs, end of odd runs).
    - `"z-lines"`: each run drawn forwards, followed by a diagonal back to the next run's start.
    - `"serpentine"`: alternating runs joined by turns. Run ends are inset by spacing/2 so the quadratic-Bézier turns can bulge back out to the edge.
    - In all three, a path is split wherever a connecting move would leave the region (checked against the mask, dilated by 1 px).
  - `FILL_STYLE="lines"`: one 2-point stroke per run. The GUI's choices come from `FILL_STYLES`.
  - `FILL_DIRECTION="shape"`: `region_axis` uses image moments, and only regions with elongation ≥ `FILL_SHAPE_MIN_ELONGATION` follow their own axis. Layer angles then become relative to the first layer.
  - Jitter uses stdlib `random.Random(FILL_SEED)`, not numpy.random, whose `_pcg64` DLL is blocked by Application Control here.
- **lineart**: the input is already a line drawing (the Kontext output). It works at `LINEART_RESOLUTION_PX`, not `WORK_RESOLUTION_PX`.
  - Steps: threshold dark ink → `thin` → `remove_staircases` → `prune_spurs` → `skeleton_strokes` → `smooth_stroke`.
  - `remove_staircases` matters: L-shaped steps left by thinning give pixels 3 neighbours, which would look like junctions and cut lines into pieces.
  - `skeleton_strokes` cuts the skeleton at junctions into chains, then pairs chain ends at each junction by straightness, so each drawn line becomes one stroke.
  - With `LINEART_FILL_THICK`, ink wider than about 2 pen widths also gets nested outlines one `LINEART_PEN_MM` apart (pupils, lashes, eyebrows).
  - It uses its own simplify tolerance and minimum length, and the preview draws at the pen width.
  - **Pen colours**: `colour_labels()` blurs and smooths for photo areas, which erases thin coloured lines (they all came out black). Line art therefore uses `lineart_colour_labels()`: each ink pixel keeps its own pen, pale anti-aliased edges take the colour of the line core, and other pixels take the pen of the nearest ink pixel.
  - `COLOUR_MIN_RUN_PX` is scaled from `WORK_RESOLUTION_PX` to the line-art resolution, so a line crossing another doesn't take its colour there.
  - `LINEART_COLOUR_SOURCE`: `"drawing"` (default) uses the line's own colour. `"photo"` uses the original photo, passed as `process(..., colour_source=)` (CLI `--colour-from`), with the photo hue rules; skin and blond hair go red, so it is rarely better.
  - To check fidelity, compare against the source; it covered 99.9% of the thinned ink of `images/monalisa_kontext.png`.
- **Shared back end**: scale to mm with image Y flipped to feature +Y → `approxPolyDP` simplify → drop short strokes → greedy nearest-neighbour ordering (strokes may be reversed) → URScript.

Core API, used by both the CLI and the GUI: `process(bgr, style) -> Result` → `urscript_text(result, name, colour)` / `render_preview(result, px_per_mm)`. These functions work in memory. Only `main()` writes files.

### Pen colours

The robot has up to 4 pens, defined in `PEN_COLOURS` (BGR values). `ACTIVE_COLOURS` lists the pens in use.

- **Colour map**: `colour_labels()` assigns each pixel the index of an active pen. The "hue" method sends grey or dark pixels to black and coloured pixels to the nearest pen hue. The "nearest" method uses Lab distance with lightness down-weighted.
- **Paper fill**: light, unsaturated "paper" pixels inherit the label of the nearest non-paper pixel, so outlines take the colour of the object rather than the white background.
- **Splitting strokes**: traced strokes are cut where the colour changes (`split_by_colour`, short runs merged). Hatch runs are split on (tone mask, pen) together.
- **Result**: `Result.layers` maps each pen to its ordered strokes, and one script is emitted per pen that has strokes (`<name>_<colour>.script`). The CLI without `--colours` stays single-pen black and writes `<name>.script`.

### GUI (`robot_painter_gui.py`)

- **Settings**: the controls are declared in the `TABS` table, mapping each core attribute to its widget. `apply_settings()` copies the widget values onto the `image_to_urscript` module globals only when a job starts, so the worker thread never sees settings change mid-run. A new module constant needs a `TABS` entry to appear in the GUI.
- **Processing**: one worker thread at a time. Changes made while it runs set `pending`, which re-runs the job once it finishes. Results come back through a queue polled by `after()`, because Tk may only be touched from the main thread.
- **Webcam**: a `Webcam` thread reads frames continuously, and `webcam_tick` processes the latest one every N seconds.
- **Saving**: Save writes the `self.job` snapshot, i.e. the exact source frame that produced the preview on screen.
- **High-DPI**: the app calls `SetProcessDpiAwareness(1)`. Screenshot tests also need DPI awareness, or `ImageGrab` coordinates don't match.

### AI restyle (`diffusion_worker.py`)

This optional pre-step runs Stable Diffusion 1.5 img2img with a lineart ControlNet (`lllyasviel/control_v11p_sd15_lineart`). Its output image replaces the source before the normal pipeline.

- **Separate process**: the worker runs in its own uv script environment: Python <3.13, torch from the cu128 index (needed for the RTX 5070 / Blackwell), diffusers and transformers. That keeps PyTorch out of the GUI environment.
- **Protocol**: the GUI's `DiffusionClient` starts it with `uv run src/diffusion_worker.py`. They talk in JSON lines over stdin/stdout, passing images as temp PNGs. The worker's stderr goes to a `worker.log` in a temp dir.
- **Control image**: made in the worker from DoG lines, not with `controlnet_aux` (that would be another heavy dependency).
- **Weights**: models are cached in `~/.cache/huggingface`, about 3.5 GB. Load with `use_safetensors=True`. A flaky download once fell back silently to pickle `.bin` weights.
- **Speed**: the first model load took about 8 minutes, including the download. One restyle takes about 2.2 s warm at 512 px.
- **Caching**: the GUI caches the AI image per `(source_version, AI params)`, so changing drawing settings doesn't re-run diffusion.
- **Self-test**: `uv run src/diffusion_worker.py --selftest images/monalisa.png` writes `images/monalisa_ai.png`.

#### Second engine: FLUX.1 Kontext via Nunchaku (`kontext_worker.py`)

The AI tab has a **Model** dropdown (`AI_ENGINES` in the GUI). Each engine has its own worker script, settings panel and defaults. Only one worker runs at a time: switching engines stops the other worker to free GPU memory.

- **What it does**: FLUX.1 Kontext [dev] is an instruction-following image editor. It redraws the photo as line art from a text prompt and needs no control image.
- **Same protocol**: it speaks the same protocol as `diffusion_worker.py` and reuses its `serve()`, `fit_size()` and I/O helpers.
- **Environment**: Python 3.12, torch 2.9 from cu128, and the Nunchaku 1.2.1 wheel pinned by URL. The wheel tag must match the torch version.
- **Precision**: `get_precision()` picks int4 (Ada, the RTX 4060 Ti 16 GB on this machine) or fp4 (Blackwell).
- **Weights**: about 9 GB of non-gated 4-bit transformer and T5 from `nunchaku-ai/`. CLIP, VAE and tokenizers come from the **gated** `black-forest-labs/FLUX.1-Kontext-dev`. Accept the licence on its Hugging Face page and run `uvx hf auth login` once. The licence is non-commercial.
- **Self-test**: `uv run src/kontext_worker.py --selftest images/monalisa.png` writes `images/monalisa_kontext.png`.
- **Output size**: the diffusers pipeline rescales its output to `max_area` (default 1 MP) while the input stays at the requested size. That offset the drawing against the photo, and made the Size setting useless. The worker therefore passes `max_area = h*w`.
- **Speed**: about 35–40 s per image at 1024 px on the 4060 Ti; worker start-up takes 12–20 s once cached.
- **Colour prompts**: Kontext ignores palette lists such as "use four pen colours (black, green, red, blue)" and draws black. It does draw coloured lines when each colour is tied to an object ("red outlines on the tulips, green on the stems, black elsewhere"). Generic wording like "outline red objects in red" gave coloured *fills* and black tulips.
- **Style lock**: the output is already line art. While Kontext is on, the GUI locks the style to `lineart` and greys out the Edges, Comic and Fill tabs (`sync_style_lock`). The lock is released, and the user's own style restored, when Kontext is turned off or its worker fails.

The preview PNG is large (~1000 px). Hatching aliases badly when it is viewed scaled down, so crop it before judging the hatch layers.

The generated script defines `pen_to`, `line_to` and `pen_up` helpers and emits one call per point. Blend radii are capped at 0.45× the adjacent segment lengths, because UR rejects overlapping blends.

## URScript conventions used

- All motion is expressed relative to a PolyScope **feature** named `Feature1`, which must be defined in the robot installation (it is the drawing plane / canvas frame). Poses are built with `pose_trans(myFeature, p[x, y, z, rx, ry, rz])`, so `x`, `y`, `z` are in the feature's local frame, in **metres**.
- Tool orientation is held constant via the rotation-vector values `a`, `b`, `c` (`1.6, -1.4, -0.8`). These were tuned by hand for the current end effector. The script logs the starting TCP pose in both base and feature-local frames (`textmsg`) so the orientation can be re-derived if the tool changes.
- Note the name clash: `a` is both the orientation component and the `a=` (acceleration) keyword argument of `movel`. The keyword argument is passed `acc`, not the variable `a`.
- Linear moves use `movel` with `speed = 0.05` m/s and `acc = 0.2` m/s². The current script traces a 100 mm square (`d = 0.100`) in the feature's XY plane at z = 0.
