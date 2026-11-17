# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "tkinterdnd2"]
# ///
"""
Tk GUI for image_to_urscript.py.

    uv run src/robot_painter_gui.py

- Drag & drop an image onto the window (or click the source panel / Ctrl+O)
- All conversion settings are live: the preview regenerates when one changes
- "Live webcam" re-captures and re-processes a frame every N seconds
- "Save" (Ctrl+S) writes the source image, preview, comic image and .script
- "AI" tab: optional AI restyle before drawing, run in a separate process with its own
  heavy dependencies. Two engines: Stable Diffusion 1.5 + lineart ControlNet
  (diffusion_worker.py) or FLUX.1 Kontext, 4-bit via Nunchaku (kontext_worker.py)
"""

import base64
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

import image_to_urscript as core

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAVE_DND = True
except Exception:  # library missing or its DLL blocked -> click-to-open still works
    HAVE_DND = False

IMAGE_TYPES = [("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")]

# (core attribute, label, kind, min, max, step)
TABS = {
    "Drawing": [
        ("IMAGE_WIDTH_MM", "Width (mm)", "float", 10, 1000, 5),
        ("IMAGE_MAX_HEIGHT_MM", "Max height (mm)", "float", 10, 1000, 5),
        ("OFFSET_X_MM", "Offset X (mm)", "float", -500, 500, 5),
        ("OFFSET_Y_MM", "Offset Y (mm)", "float", -500, 500, 5),
        ("FLIP_X", "Flip X", "bool"),
        ("FLIP_Y", "Flip Y", "bool"),
        ("WORK_RESOLUTION_PX", "Work resolution (px)", "int", 100, 2000, 50),
        ("SIMPLIFY_TOLERANCE_MM", "Simplify tolerance (mm)", "float", 0, 5, 0.1),
        ("MIN_LINE_LENGTH_MM", "Min line length (mm)", "float", 0, 50, 0.5),
    ],
    "Robot": [
        ("PEN_DOWN_Z_MM", "Pen down Z (mm)", "float", -100, 100, 0.5),
        ("PEN_UP_Z_MM", "Pen up Z (mm)", "float", -100, 100, 1),
        ("ORIENT_RX", "Orientation RX", "float", -6.3, 6.3, 0.05),
        ("ORIENT_RY", "Orientation RY", "float", -6.3, 6.3, 0.05),
        ("ORIENT_RZ", "Orientation RZ", "float", -6.3, 6.3, 0.05),
        ("DRAW_SPEED", "Draw speed (m/s)", "float", 0.001, 1, 0.01),
        ("TRAVEL_SPEED", "Travel speed (m/s)", "float", 0.001, 1, 0.01),
        ("ACCELERATION", "Acceleration (m/s²)", "float", 0.01, 5, 0.05),
        ("BLEND_RADIUS_MM", "Blend radius (mm)", "float", 0, 10, 0.1),
    ],
    "Colour": [
        ("COLOUR_MIN_SATURATION", "Min saturation (hue)", "int", 0, 255, 5),
        ("COLOUR_DARK_VALUE", "Dark → black below", "int", 0, 255, 5),
        ("COLOUR_PAPER_VALUE", "Paper (white) above", "int", 0, 256, 5),
        ("COLOUR_SMOOTH_PX", "Colour map smoothing (px)", "int", 1, 51, 2),
        ("COLOUR_MIN_RUN_PX", "Min colour run on a line (px)", "int", 1, 100, 1),
    ],
    "Edges": [
        ("BLUR_KERNEL", "Blur kernel (odd)", "int", 1, 31, 2),
        ("CANNY_LOW", "Canny low", "int", 0, 500, 5),
        ("CANNY_HIGH", "Canny high", "int", 0, 500, 5),
    ],
    "Line art": [
        ("LINEART_RESOLUTION_PX", "Work resolution (px)", "int", 300, 3000, 100),
        ("LINEART_THRESHOLD", "Ink darker than (0-255)", "int", 1, 254, 5),
        ("LINEART_BRIDGE_PX", "Bridge gaps (px)", "int", 0, 5, 1),
        ("LINEART_MIN_AREA_PX", "Min speck (px)", "int", 0, 500, 5),
        ("LINEART_SPUR_PX", "Prune spurs (px)", "int", 0, 30, 1),
        ("LINEART_SMOOTH_PX", "Smooth lines (px)", "int", 0, 10, 1),
        ("LINEART_SIMPLIFY_MM", "Simplify tolerance (mm)", "float", 0, 2, 0.05),
        ("LINEART_MIN_LENGTH_MM", "Min line length (mm)", "float", 0, 20, 0.1),
        ("LINEART_COLOUR_SOURCE", "Colours from (multi-pen)", "choice", core.LINEART_COLOUR_SOURCES),
        ("LINEART_FILL_THICK", "Fill thick ink", "bool"),
        ("LINEART_PEN_MM", "Pen width (mm)", "float", 0.1, 3, 0.05),
    ],
    "Comic": [
        ("COMIC_INK", "Draw ink lines", "bool"),
        ("COMIC_HATCH", "Draw shading (Fill tab)", "bool"),
        ("EQUALIZE", "Equalise contrast", "bool"),
        ("SMOOTH_ITERATIONS", "Tone smoothing passes", "int", 0, 10, 1),
        ("SMOOTH_SIGMA_COLOR", "Smoothing strength", "float", 1, 200, 5),
        ("TONE_LEVELS", "Tone levels", "int", 2, 8, 1),
        ("INK_SMOOTH_ITERATIONS", "Ink smoothing passes", "int", 0, 10, 1),
        ("INK_SIGMA", "Ink line scale (px)", "float", 0.3, 5, 0.1),
        ("INK_THRESHOLD", "Ink threshold", "float", 0.001, 0.1, 0.001),
        ("INK_MIN_AREA_PX", "Ink min speck (px)", "int", 0, 500, 5),
    ],
    "Fill": [
        ("FILL_STYLE", "Fill style", "choice", core.FILL_STYLES),
        ("FILL_DIRECTION", "Direction", "choice", ["shape", "fixed"]),
        ("FILL_ROUND_TURNS", "Rounded turns (serpentine)", "bool"),
        ("FILL_JITTER_MM", "Hand-drawn jitter (mm)", "float", 0, 3, 0.1),
        ("FILL_SHAPE_MIN_ELONGATION", "Follow shape if elongated ≥", "float", 1, 10, 0.25),
        ("FILL_SEED", "Jitter seed", "int", 0, 9999, 1),
        ("HATCH_MIN_REGION_PX", "Min region (px)", "int", 0, 2000, 10),
    ],
}

COLOUR_METHODS = {"hue": "Hue (paintings, photos)", "nearest": "Nearest colour (flat graphics)"}

STYLES = {"edges": "Edges (outlines)", "comic": "Comic (ink + shading)",
          "lineart": "Line art (trace a line drawing)"}
# settings tabs that only apply to one style; greyed out while the style is locked to line art
STYLE_TABS = {"Edges", "Comic", "Fill"}

MAX_HATCH_LAYERS = 4
EXTRA_HATCH_DEFAULTS = [(0, 3.0, 0), (90, 3.0, 0)]  # offered as disabled extra layers


def to_photo(img: np.ndarray, max_w: int, max_h: int) -> tk.PhotoImage:
    """Fit a BGR/gray numpy image into max_w x max_h and convert to a Tk PhotoImage."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    if scale < 1:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return tk.PhotoImage(data=base64.b64encode(buf.tobytes()).decode("ascii"), format="png")


class Webcam:
    """Reads frames continuously in a thread so `latest()` is always the current frame."""

    def __init__(self, index: int):
        self.index = index
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.error = f"Could not open camera {self.index}"
            return
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                with self._lock:
                    self._frame = frame
        finally:
            cap.release()

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stop.set()


# AI restyle engines. Each runs its own worker script (same JSON-lines protocol).
# "text": prompt fields, "numbers": (key, label, min, max, step, is_int) spinboxes.
AI_ENGINES = {
    "sd15": {
        "label": "Stable Diffusion 1.5 + lineart ControlNet",
        "script": "diffusion_worker.py",
        "note": "img2img guided by the photo's lines.\nFirst use downloads ~3 GB of model weights.",
        "text": (("prompt", "Prompt"), ("negative", "Negative prompt")),
        "numbers": (("strength", "Strength (0 = keep, 1 = repaint)", 0.05, 1.0, 0.05, False),
                    ("control_scale", "Lineart guidance", 0.0, 2.0, 0.1, False),
                    ("steps", "Steps", 4, 50, 1, True),
                    ("guidance", "Prompt guidance", 1.0, 15.0, 0.5, False),
                    ("seed", "Seed", 0, 999999, 1, True)),
        "defaults": {
            "prompt": "comic book illustration, bold black ink outlines, flat colours, clean line art",
            "negative": "blurry, photo, noisy, grainy, text, watermark, signature",
            "strength": 0.55, "steps": 20, "guidance": 7.0, "control_scale": 1.0, "seed": 42,
        },
    },
    "kontext": {
        "label": "FLUX.1 Kontext (Nunchaku 4-bit)",
        "script": "kontext_worker.py",
        "note": "Redraws the photo as the instruction says.\nFirst use downloads ~10 GB and "
                "needs a Hugging Face\nlogin (see kontext_worker.py). Drawn as line art.\n"
                "For colour pens, name a colour per object, e.g.\n\"red lines for the tulips, "
                "green for stems,\nblack for everything else\".",
        "text": (("prompt", "Instruction"),),
        "numbers": (("size", "Size (longest side, px)", 512, 1536, 64, True),
                    ("steps", "Steps", 4, 50, 1, True),
                    ("guidance", "Guidance", 1.0, 10.0, 0.5, False),
                    ("seed", "Seed", 0, 999999, 1, True)),
        "defaults": {
            "prompt": ("Convert this photo into a clean black and white line drawing, colouring "
                       "book style. Thin uniform black outlines on a plain white background. "
                       "No shading, no hatching, no grey, no colour fill. Keep the faces and pose."),
            "size": 1024, "steps": 20, "guidance": 2.5, "seed": 42,
        },
    },
}
AI_ENGINE_BY_LABEL = {e["label"]: key for key, e in AI_ENGINES.items()}


class DiffusionClient:
    """Runs diffusion_worker.py as a child process (via uv, so torch etc. stay out of the
    GUI environment) and sends it one image at a time. The model is loaded once."""

    def __init__(self, engine: str):
        self.engine = engine
        script = Path(__file__).with_name(AI_ENGINES[engine]["script"])
        self.state = "starting"       # starting -> ready | failed ; ready -> stopped
        self.error = None
        self.info = ""
        self.tmp = Path(tempfile.mkdtemp(prefix="robotpainter_ai_"))
        self.log_path = self.tmp / "worker.log"
        self._lock = threading.Lock()
        self._replies = queue.Queue()
        self._n = 0
        uv = shutil.which("uv")
        if not uv:
            self.state, self.error = "failed", "'uv' not found on PATH"
            self.proc = None
            return
        self._log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [uv, "run", str(script)], cwd=script.parent,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "ready" in msg:
                if msg["ready"]:
                    self.state = "ready"
                    self.info = f"{msg.get('device')}, loaded in {msg.get('load_s')} s"
                else:
                    self.state, self.error = "failed", msg.get("error")
            else:
                self._replies.put(msg)
        if self.state == "starting":
            self.state, self.error = "failed", self.error or "worker exited during start-up"
        elif self.state == "ready":
            self.state = "stopped"
        self._replies.put({"ok": False, "error": "AI worker stopped"})

    def last_log_line(self) -> str:
        """Most recent progress line (download/loading progress uses carriage returns)."""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        parts = [p.strip() for p in text.replace("\r", "\n").split("\n") if p.strip()]
        return parts[-1][:110] if parts else ""

    def restyle(self, bgr: np.ndarray, params: dict) -> tuple[np.ndarray, float]:
        """Blocking: returns (restyled BGR image, seconds). Call from a worker thread."""
        with self._lock:
            if self.state != "ready":
                raise RuntimeError(f"AI model not ready ({self.state})")
            self._n += 1
            src, dst = self.tmp / f"in_{self._n}.png", self.tmp / f"out_{self._n}.png"
            core.save_image(src, bgr)
            self.proc.stdin.write(json.dumps({**params, "in": str(src), "out": str(dst)}) + "\n")
            self.proc.stdin.flush()
            msg = self._replies.get(timeout=600)
            src.unlink(missing_ok=True)
            if not msg.get("ok"):
                raise RuntimeError(msg.get("error", "AI restyle failed"))
            img = core.read_image(dst)
            dst.unlink(missing_ok=True)
            return img, float(msg.get("seconds", 0))

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()  # worker leaves its request loop and exits
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Robot Painter")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{int(sw * 0.8)}x{int(sh * 0.8)}+{int(sw * 0.1)}+{int(sh * 0.05)}")
        root.minsize(int(sw * 0.5), int(sh * 0.5))
        self.closed = False

        self.vars: dict[str, tk.Variable] = {}
        self.defaults = {attr: getattr(core, attr) for specs in TABS.values() for attr, *_ in specs}
        self.default_layers = list(core.HATCH_LAYERS)

        self.source = None          # BGR image currently loaded
        self.source_name = None
        self.source_version = 0     # bumps on every new image / webcam frame
        self.job = None             # (source, name, result, scripts, elapsed, ai_img) on screen
        self.ai = None              # DiffusionClient for the selected engine, started on enable
        self.ai_cache = None        # (key, image): skip diffusion when only drawing settings change
        self.waiting_for_ai = False
        self.busy = False
        self.pending = False
        self.results = queue.Queue()
        self.webcam = None
        self._debounce = None
        self._resize = None

        self._build_controls()
        self._build_views()
        root.bind("<Control-o>", lambda e: self.open_dialog())
        root.bind("<Control-s>", lambda e: self.save())
        root.protocol("WM_DELETE_WINDOW", self.close)
        if HAVE_DND:
            root.drop_target_register(DND_FILES)
            root.dnd_bind("<<Drop>>", self.on_drop)
        self.root.after(100, self.poll_results)

    # ------------------------------------------------------------- layout
    def _build_controls(self):
        left = ttk.Frame(self.root, padding=8)
        left.pack(side="left", fill="y")

        top = ttk.Frame(left)
        top.pack(fill="x")
        ttk.Button(top, text="Open image…", command=self.open_dialog).pack(side="left")
        ttk.Button(top, text="Reset settings", command=self.reset).pack(side="left", padx=6)

        style = ttk.LabelFrame(left, text="Style", padding=6)
        style.pack(fill="x", pady=(8, 0))
        self.style = tk.StringVar(value="comic")
        self.user_style = "comic"   # restored when the AI line-art lock is released
        self.style_buttons = {}
        for value, text in STYLES.items():
            rb = ttk.Radiobutton(style, text=text, value=value, variable=self.style,
                                 command=self.on_style)
            rb.pack(anchor="w")
            self.style_buttons[value] = rb
        self.style_lock_note = ttk.Label(style, text="Locked to line art: the AI drawing is "
                                         "traced line for line", foreground="#666", wraplength=280)

        self._build_pens(left)

        nb = ttk.Notebook(left)
        nb.pack(fill="both", expand=True, pady=8)
        self.nb = nb
        self.tab_frames = {}
        for tab, specs in TABS.items():
            frame = ttk.Frame(nb, padding=6)
            nb.add(frame, text=tab)
            self.tab_frames[tab] = frame
            offset = 0
            if tab == "Colour":
                ttk.Label(frame, text="Method").grid(row=0, column=0, sticky="w")
                self.colour_method = tk.StringVar(value=COLOUR_METHODS[core.COLOUR_METHOD])
                cb = ttk.Combobox(frame, textvariable=self.colour_method, state="readonly",
                                  values=list(COLOUR_METHODS.values()), width=26)
                cb.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 6))
                cb.bind("<<ComboboxSelected>>", lambda e: self.request_update())
                offset = 2
            for row, spec in enumerate(specs, offset):
                self._add_control(frame, row, *spec)
            if tab == "Fill":
                self._build_hatch_layers(frame, len(specs))
        self._build_ai_tab(nb)

        cam = ttk.LabelFrame(left, text="Webcam", padding=6)
        cam.pack(fill="x")
        self.live = tk.BooleanVar(value=False)
        ttk.Checkbutton(cam, text="Live webcam", variable=self.live,
                        command=self.toggle_webcam).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(cam, text="Camera index").grid(row=1, column=0, sticky="w")
        self.cam_index = tk.IntVar(value=0)
        ttk.Spinbox(cam, from_=0, to=9, textvariable=self.cam_index, width=6).grid(row=1, column=1, sticky="w")
        ttk.Label(cam, text="Update every (s)").grid(row=2, column=0, sticky="w")
        self.cam_interval = tk.DoubleVar(value=5)
        ttk.Spinbox(cam, from_=1, to=120, increment=1, textvariable=self.cam_interval,
                    width=6).grid(row=2, column=1, sticky="w")

        out = ttk.LabelFrame(left, text="Save", padding=6)
        out.pack(fill="x", pady=(8, 0))
        self.out_dir = tk.StringVar(value=str(Path.cwd() / "output"))
        ttk.Entry(out, textvariable=self.out_dir, width=28).grid(row=0, column=0, sticky="ew")
        ttk.Button(out, text="…", width=3, command=self.choose_dir).grid(row=0, column=1, padx=(4, 0))
        ttk.Button(out, text="Save image + preview + script  (Ctrl+S)",
                   command=self.save).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

    def _build_ai_tab(self, nb):
        frame = ttk.Frame(nb, padding=6)
        nb.add(frame, text="AI")
        self.ai_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="AI restyle before drawing", variable=self.ai_enabled,
                        command=self.toggle_ai).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(frame, text="Model").grid(row=1, column=0, sticky="w")
        self.ai_engine = tk.StringVar(value=AI_ENGINES["sd15"]["label"])
        cb = ttk.Combobox(frame, textvariable=self.ai_engine, state="readonly", width=34,
                          values=[e["label"] for e in AI_ENGINES.values()])
        cb.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        cb.bind("<<ComboboxSelected>>", lambda e: self.toggle_ai())

        # one settings panel per engine; only the selected one is shown
        self.ai_vars = {}     # engine -> {key: tk variable}
        self.ai_panels = {}
        for engine, spec in AI_ENGINES.items():
            panel = ttk.Frame(frame)
            panel.grid(row=3, column=0, columnspan=2, sticky="ew")
            panel.columnconfigure(0, weight=1)
            self.ai_panels[engine] = panel
            vars_ = self.ai_vars[engine] = {}
            defaults = spec["defaults"]
            ttk.Label(panel, text=spec["note"], foreground="#666").grid(
                row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
            row = 1
            for key, label in spec["text"]:
                ttk.Label(panel, text=label).grid(row=row, column=0, columnspan=2, sticky="w")
                var = tk.StringVar(value=defaults[key])
                entry = ttk.Entry(panel, textvariable=var, width=34)
                entry.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(0, 4))
                # text fields only re-run on Enter / leaving the field, not on every keystroke
                entry.bind("<Return>", lambda e: self.request_update())
                entry.bind("<FocusOut>", lambda e: self.request_update())
                vars_[key] = var
                row += 2
            for key, label, lo, hi, step, is_int in spec["numbers"]:
                var = tk.IntVar(value=defaults[key]) if is_int else tk.DoubleVar(value=defaults[key])
                ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", pady=1)
                ttk.Spinbox(panel, from_=lo, to=hi, increment=step, textvariable=var,
                            width=9).grid(row=row, column=1, sticky="e", pady=1)
                var.trace_add("write", lambda *_: self.request_update())
                vars_[key] = var
                row += 1
        self.ai_status = ttk.Label(frame, text="AI off", foreground="#555", wraplength=300)
        self.ai_status.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.show_ai_panel()

    def selected_engine(self) -> str:
        return AI_ENGINE_BY_LABEL.get(self.ai_engine.get(), "sd15")

    def show_ai_panel(self):
        for engine, panel in self.ai_panels.items():
            if engine == self.selected_engine():
                panel.grid()
            else:
                panel.grid_remove()

    def toggle_ai(self):
        """AI checkbox or model choice changed: start the worker for the selected engine.
        Only one worker runs at a time, so switching frees the other model's GPU memory."""
        engine = self.selected_engine()
        self.show_ai_panel()
        if self.ai is not None and self.ai.engine != engine:
            self.ai.stop()
            self.ai = None
        if self.ai_enabled.get() and (self.ai is None or self.ai.state in ("failed", "stopped")):
            self.ai = DiffusionClient(engine)
        if not (self.job and self.job[5] is not None):
            self.show_ai_image(None)
        self.sync_style_lock()
        self.request_update()

    def on_style(self):
        self.user_style = self.style.get()
        self.request_update()

    def line_art_locked(self) -> bool:
        """Kontext already draws the line drawing, so the pipeline must only trace it.
        Released if its worker failed, since the plain photo is drawn instead then."""
        return (self.ai_enabled.get() and self.selected_engine() == "kontext"
                and not (self.ai and self.ai.state in ("failed", "stopped")))

    def sync_style_lock(self):
        """Force the line-art style while Kontext is on, and grey out the other styles and
        their settings tabs. Restores the user's own style choice afterwards."""
        locked = self.line_art_locked()
        if locked == getattr(self, "_style_locked", False):
            return
        self._style_locked = locked
        self.style.set("lineart" if locked else self.user_style)
        for value, rb in self.style_buttons.items():
            rb.configure(state="disabled" if locked and value != "lineart" else "normal")
        for tab in STYLE_TABS:
            self.nb.tab(self.tab_frames[tab], state="disabled" if locked else "normal")
        if locked:
            self.style_lock_note.pack(anchor="w", pady=(4, 0))
            if self.nb.tab(self.nb.select(), "text") in STYLE_TABS:
                self.nb.select(self.tab_frames["Line art"])
        else:
            self.style_lock_note.pack_forget()
        self.request_update()

    def ai_params(self) -> dict | None:
        """Current settings of the selected AI engine, or None if AI restyle is off."""
        if not self.ai_enabled.get():
            return None
        engine = self.selected_engine()
        params = {}
        for key, var in self.ai_vars[engine].items():
            try:
                params[key] = var.get()
            except tk.TclError:
                params[key] = AI_ENGINES[engine]["defaults"][key]
        return params

    def update_ai_status(self):
        if not self.ai_enabled.get():
            text = "AI off" + (" (model stays loaded)" if self.ai and self.ai.state == "ready" else "")
        elif self.ai is None:
            text = "AI off"
        elif self.ai.state == "starting":
            text = f"Starting AI worker… {self.ai.last_log_line()}"
        elif self.ai.state == "ready":
            text = f"AI ready: {AI_ENGINES[self.ai.engine]['label']} ({self.ai.info})"
        else:
            text = f"AI unavailable: {self.ai.error}\nLog: {self.ai.log_path}"
        if self.ai_status.cget("text") != text:
            self.ai_status.configure(text=text)

    def _build_pens(self, parent):
        """Toggle buttons for the robot's pens; a pressed (coloured) button = pen in use."""
        box = ttk.LabelFrame(parent, text="Pen colours (one script per colour)", padding=6)
        box.pack(fill="x", pady=(8, 0))
        self.pen_vars = {}
        for i, (name, (b, g, r)) in enumerate(core.PEN_COLOURS.items()):
            var = tk.BooleanVar(value=True)
            hex_colour = f"#{r:02x}{g:02x}{b:02x}"
            tk.Checkbutton(box, text=name.capitalize(), variable=var, indicatoron=False,
                           width=7, pady=3, fg="white", bg="#9a9a9a", selectcolor=hex_colour,
                           activebackground=hex_colour, activeforeground="white",
                           font=("Segoe UI", 10, "bold"),
                           command=self.request_update).grid(row=0, column=i, padx=2)
            self.pen_vars[name] = var

    def _add_control(self, parent, row, attr, label, kind, lo=None, hi=None, step=None):
        value = getattr(core, attr)
        if kind == "bool":
            var = tk.BooleanVar(value=value)
            ttk.Checkbutton(parent, text=label, variable=var).grid(row=row, column=0, columnspan=2, sticky="w")
        elif kind == "choice":  # `lo` holds the list of allowed values
            var = tk.StringVar(value=value)
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
            ttk.Combobox(parent, textvariable=var, values=lo, state="readonly",
                         width=8).grid(row=row, column=1, sticky="e", pady=1)
        else:
            var = tk.IntVar(value=value) if kind == "int" else tk.DoubleVar(value=value)
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
            ttk.Spinbox(parent, from_=lo, to=hi, increment=step, textvariable=var,
                        width=9).grid(row=row, column=1, sticky="e", pady=1)
        var.trace_add("write", lambda *_: self.request_update())
        self.vars[attr] = var

    def _build_hatch_layers(self, parent, row):
        box = ttk.LabelFrame(parent, text="Fill layers (tone 0 = darkest)", padding=4)
        box.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for c, text in enumerate(("On", "Angle °", "Spacing mm", "Max tone")):
            ttk.Label(box, text=text).grid(row=0, column=c, padx=2)
        layers = list(core.HATCH_LAYERS)
        enabled = [True] * len(layers)
        for extra in EXTRA_HATCH_DEFAULTS:
            if len(layers) < MAX_HATCH_LAYERS:
                layers.append(extra)
                enabled.append(False)
        self.hatch_vars = []
        for i, ((angle, spacing, max_tone), on) in enumerate(zip(layers, enabled), 1):
            vs = (tk.BooleanVar(value=on), tk.DoubleVar(value=angle),
                  tk.DoubleVar(value=spacing), tk.IntVar(value=max_tone))
            ttk.Checkbutton(box, variable=vs[0]).grid(row=i, column=0)
            ttk.Spinbox(box, from_=-180, to=180, increment=15, textvariable=vs[1], width=6).grid(row=i, column=1)
            ttk.Spinbox(box, from_=0.3, to=20, increment=0.5, textvariable=vs[2], width=6).grid(row=i, column=2)
            ttk.Spinbox(box, from_=0, to=7, increment=1, textvariable=vs[3], width=4).grid(row=i, column=3)
            for v in vs:
                v.trace_add("write", lambda *_: self.request_update())
            self.hatch_vars.append(vs)

    def _build_views(self):
        right = ttk.Frame(self.root, padding=8)
        right.pack(side="left", fill="both", expand=True)
        for col in range(3):  # source | AI-styled source | robot drawing, equal widths
            right.columnconfigure(col, weight=1, uniform="v")
        right.rowconfigure(1, weight=1)

        ttk.Label(right, text="Source").grid(row=0, column=0, sticky="w")
        ttk.Label(right, text="Source (AI styled)").grid(row=0, column=1, sticky="w")
        ttk.Label(right, text="Robot drawing preview").grid(row=0, column=2, sticky="w")

        self.source_panel = tk.Frame(right, bg="#e8e8e8", highlightthickness=2,
                                     highlightbackground="#b0b0b0")
        self.source_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.source_panel.grid_propagate(False)
        drop_hint = "Drop an image here\nor click to open" if HAVE_DND else "Click to open an image\n(drag & drop unavailable)"
        self.source_label = tk.Label(self.source_panel, text=drop_hint, bg="#e8e8e8",
                                     fg="#555", font=("Segoe UI", 13), cursor="hand2")
        self.source_label.place(relx=0.5, rely=0.5, anchor="center")
        for w in (self.source_panel, self.source_label):
            w.bind("<Button-1>", lambda e: self.open_dialog())

        self.ai_panel = tk.Frame(right, bg="#f4f4f4", highlightthickness=1,
                                 highlightbackground="#b0b0b0")
        self.ai_panel.grid(row=1, column=1, sticky="nsew", padx=(0, 6))
        self.ai_panel.grid_propagate(False)
        self.ai_label = tk.Label(self.ai_panel, text="AI restyle off\n(AI tab)", bg="#f4f4f4",
                                 fg="#777", font=("Segoe UI", 11))
        self.ai_label.place(relx=0.5, rely=0.5, anchor="center")

        self.preview_panel = tk.Frame(right, bg="white", highlightthickness=1,
                                      highlightbackground="#b0b0b0")
        self.preview_panel.grid(row=1, column=2, sticky="nsew")
        self.preview_panel.grid_propagate(False)
        self.preview_label = tk.Label(self.preview_panel, bg="white")
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")

        info = ttk.Frame(right)
        info.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.comic_label = tk.Label(info)
        self.comic_label.pack(side="left", padx=(0, 6))
        self.colour_label = tk.Label(info)
        self.colour_label.pack(side="left", padx=(0, 12))
        self.stats = ttk.Label(info, text="No image loaded", font=("Consolas", 11), justify="left")
        self.stats.pack(side="left", anchor="n")
        self.colour_rows = ttk.Frame(info)
        self.colour_rows.pack(side="left", anchor="n", padx=(24, 0))

        self.status = ttk.Label(right, text="", foreground="#555")
        self.status.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))

        for panel in (self.source_panel, self.ai_panel, self.preview_panel):
            panel.bind("<Configure>", self.on_resize)

    # ------------------------------------------------------------ sources
    def open_dialog(self):
        path = filedialog.askopenfilename(filetypes=IMAGE_TYPES)
        if path:
            self.load_file(Path(path))

    def on_drop(self, event):
        paths = self.root.tk.splitlist(event.data)
        if paths:
            self.load_file(Path(paths[0]))

    def load_file(self, path: Path):
        try:
            img = core.read_image(path)
        except ValueError as e:
            messagebox.showerror("Robot Painter", str(e))
            return
        if self.live.get():
            self.live.set(False)
            self.toggle_webcam()
        self.set_source(img, path.stem)

    def set_source(self, img, name):
        self.source, self.source_name = img, name
        self.source_version += 1
        self.show_source(img)
        if self.ai_enabled.get():
            self.show_ai_image(None)  # "AI restyling…" until the new AI image is ready
        self.start_job()

    def show_source(self, img):
        p = self.source_panel
        photo = to_photo(img, max(p.winfo_width() - 8, 50), max(p.winfo_height() - 8, 50))
        self.source_label.configure(image=photo, text="")
        self.source_label.image = photo

    def toggle_webcam(self):
        if self.live.get():
            self.webcam = Webcam(self.cam_index.get())
            self.status.configure(text="Starting webcam…")
            self.root.after(1500, self.webcam_tick)
        elif self.webcam:
            self.webcam.stop()
            self.webcam = None
            self.status.configure(text="Webcam stopped")

    def webcam_tick(self):
        cam = self.webcam
        if not self.live.get() or cam is None:
            return
        if cam.error:
            self.live.set(False)
            self.webcam = None
            messagebox.showerror("Robot Painter", cam.error)
            return
        frame = cam.latest()
        if frame is not None:
            self.set_source(frame, "webcam")
        try:
            interval = max(1.0, float(self.cam_interval.get()))
        except (tk.TclError, ValueError):
            interval = 5.0
        self.root.after(int(interval * 1000), self.webcam_tick)

    # --------------------------------------------------------- processing
    def request_update(self):
        if self._debounce:
            self.root.after_cancel(self._debounce)
        self._debounce = self.root.after(350, self.start_job)

    def apply_settings(self):
        """Copy GUI values into the core module. Called only when a job starts, so the
        worker thread never sees settings change mid-run."""
        for attr, var in self.vars.items():
            try:
                value = var.get()
            except tk.TclError:  # half-typed value: keep the previous one
                continue
            if isinstance(self.defaults[attr], int) and not isinstance(self.defaults[attr], bool):
                value = int(round(value))
            setattr(core, attr, value)
        layers = []
        for on, angle, spacing, max_tone in self.hatch_vars:
            try:
                if on.get():
                    layers.append((angle.get(), max(0.3, spacing.get()), int(max_tone.get())))
            except tk.TclError:
                continue
        core.HATCH_LAYERS = layers
        core.ACTIVE_COLOURS = [c for c, v in self.pen_vars.items() if v.get()] or ["black"]
        label_to_key = {v: k for k, v in COLOUR_METHODS.items()}
        core.COLOUR_METHOD = label_to_key.get(self.colour_method.get(), "hue")

    def start_job(self):
        self._debounce = None
        if self.source is None:
            return
        if self.busy:
            self.pending = True
            return
        ai_params = self.ai_params()
        ai_key = cached = None
        if ai_params is not None:
            if self.ai is None or self.ai.state == "starting":
                self.waiting_for_ai = True   # poll_results restarts the job once it is ready
                self.status.configure(text="Waiting for the AI model to load…")
                return
            if self.ai.state != "ready":
                ai_params = None             # failed: draw without AI, the AI tab shows why
            else:
                ai_key = (self.source_version, self.ai.engine, tuple(sorted(ai_params.items())))
                if self.ai_cache and self.ai_cache[0] == ai_key:
                    cached = self.ai_cache[1]
        self.apply_settings()
        self.busy = True
        self.status.configure(text="AI restyling…" if ai_params and cached is None else "Processing…")
        source, name, style, ai = self.source, self.source_name, self.style.get(), self.ai

        def work():
            t0 = time.perf_counter()
            try:
                ai_img = None
                if ai_params is not None:
                    ai_img = cached if cached is not None else ai.restyle(source, ai_params)[0]
                if ai_img is not None:  # line art from the AI image, colours from the photo
                    result = core.process(ai_img, style, colour_source=source)
                else:
                    result = core.process(source, style)
                scripts = {c: core.urscript_text(result, name, c)
                           for c, strokes in result.layers.items() if strokes}
                elapsed = time.perf_counter() - t0
                self.results.put(("ok", ((source, name, result, scripts, elapsed, ai_img), ai_key)))
            except Exception as e:  # report in the GUI instead of killing the thread silently
                self.results.put(("error", e))

        threading.Thread(target=work, daemon=True).start()

    def poll_results(self):
        try:
            while True:
                kind, payload = self.results.get_nowait()
                self.busy = False
                if kind == "error":
                    self.status.configure(text=f"Error: {payload}")
                else:
                    self.job, ai_key = payload
                    if ai_key is not None:
                        self.ai_cache = (ai_key, self.job[5])
                    self.show_result()
                    stamp = datetime.now().strftime("%H:%M:%S")
                    live = "  (live webcam)" if self.live.get() else ""
                    self.status.configure(text=f"Updated {stamp}{live}")
                if self.pending:
                    self.pending = False
                    self.start_job()
        except queue.Empty:
            pass
        if self.ai is not None or self.ai_enabled.get():
            self.update_ai_status()
            self.sync_style_lock()  # releases the line-art lock if the worker failed
        if self.waiting_for_ai and (self.ai is None or self.ai.state != "starting"):
            self.waiting_for_ai = False
            self.start_job()
        if not self.closed:
            self.root.after(100, self.poll_results)

    def show_result(self):
        if not self.job:
            return
        source, name, result, scripts, elapsed, ai_img = self.job
        # all three panels show the same job, so they always belong together (webcam too)
        self.show_source(source)
        self.show_ai_image(ai_img)
        p = self.preview_panel
        avail_w, avail_h = max(p.winfo_width() - 8, 50), max(p.winfo_height() - 8, 50)
        span_w = result.width_mm + abs(core.OFFSET_X_MM) + 2
        span_h = result.height_mm + abs(core.OFFSET_Y_MM) + 2
        px_per_mm = min(avail_w / span_w, avail_h / span_h)
        photo = to_photo(core.render_preview(result, px_per_mm), avail_w, avail_h)
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo

        for label, img in ((self.comic_label, result.comic), (self.colour_label, result.colour_map)):
            if img is not None:
                thumb = to_photo(img, 140, 140)
                label.configure(image=thumb)
                label.image = thumb
            else:
                label.configure(image="")
                label.image = None

        def fmt(seconds):
            mins, secs = divmod(int(seconds), 60)
            return f"{mins} min {secs:02d} s"

        h, w = source.shape[:2]
        self.stats.configure(text=(
            f"Style          {result.style}\n"
            f"Drawing size   {result.width_mm:.1f} x {result.height_mm:.1f} mm\n"
            f"Strokes        {len(result.strokes)}   ({result.n_points} points)\n"
            f"Pen-down path  {result.draw_length_mm / 1000:.2f} m\n"
            f"Est. time      ~{fmt(result.est_time_s)}  (lower bound, excl. pen changes)\n"
            f"Source         {name}  {w} x {h} px   processed in {elapsed:.2f} s"))

        # one coloured row per pen: strokes, time, script size
        for child in self.colour_rows.winfo_children():
            child.destroy()
        for i, (colour, strokes) in enumerate(result.layers.items()):
            b, g, r = core.PEN_COLOURS[colour]
            swatch = tk.Label(self.colour_rows, width=2, bg=f"#{r:02x}{g:02x}{b:02x}")
            swatch.grid(row=i, column=0, padx=(0, 6), pady=1, sticky="ns")
            text = (f"{colour:<6}{len(strokes):4d} strokes ~{fmt(result.est_times[colour])}"
                    if colour in scripts else f"{colour:<6}   – no script")
            ttk.Label(self.colour_rows, text=text, font=("Consolas", 11)).grid(row=i, column=1, sticky="w")

    def on_resize(self, _event):
        if self._resize:
            self.root.after_cancel(self._resize)

        def redraw():
            self._resize = None
            if self.job:
                self.show_result()
            elif self.source is not None:
                self.show_source(self.source)
        self._resize = self.root.after(150, redraw)

    def show_ai_image(self, img):
        """Middle panel: the AI-styled image of the current preview, or why there is none."""
        if img is None:
            text = ("AI restyle off\n(AI tab)" if not self.ai_enabled.get()
                    else "AI unavailable\n(see AI tab)" if self.ai and self.ai.state == "failed"
                    else "AI restyling…")
            self.ai_label.configure(image="", text=text)
            self.ai_label.image = None
            return
        p = self.ai_panel
        photo = to_photo(img, max(p.winfo_width() - 8, 50), max(p.winfo_height() - 8, 50))
        self.ai_label.configure(image=photo, text="")
        self.ai_label.image = photo

    # ------------------------------------------------------------- actions
    def reset(self):
        for attr, var in self.vars.items():
            var.set(self.defaults[attr])
        for i, vs in enumerate(self.hatch_vars):
            if i < len(self.default_layers):
                angle, spacing, max_tone = self.default_layers[i]
                on = True
            else:
                (angle, spacing, max_tone), on = EXTRA_HATCH_DEFAULTS[i - len(self.default_layers)], False
            for v, value in zip(vs, (on, angle, spacing, max_tone)):
                v.set(value)
        self.user_style = "comic"
        if not self.line_art_locked():
            self.style.set("comic")
        for var in self.pen_vars.values():
            var.set(True)
        self.colour_method.set(COLOUR_METHODS["hue"])
        for engine, vars_ in self.ai_vars.items():
            for key, var in vars_.items():
                var.set(AI_ENGINES[engine]["defaults"][key])
        self.request_update()

    def choose_dir(self):
        d = filedialog.askdirectory(initialdir=self.out_dir.get() or ".")
        if d:
            self.out_dir.set(d)

    def save(self):
        """Save exactly what is shown: the source frame that produced the current preview."""
        if not self.job:
            messagebox.showinfo("Robot Painter", "Nothing to save yet - load an image first.")
            return
        source, name, result, scripts, _, ai_img = self.job
        out = Path(self.out_dir.get() or ".")
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{name}_{result.style}_{datetime.now():%Y%m%d_%H%M%S}"
        images = {"source": source, "ai": ai_img, "preview": core.render_preview(result),
                  "comic": result.comic, "colours": result.colour_map}
        count = 0
        for suffix, img in images.items():
            if img is not None:
                core.save_image(out / f"{stem}_{suffix}.png", img)
                count += 1
        for colour, text in scripts.items():  # e.g. monalisa_comic_..._red.script
            (out / f"{stem}_{colour}.script").write_text(text, encoding="utf-8")
            count += 1
        self.status.configure(text=f"Saved {count} files to {out}  ({stem}_…, "
                                   f"scripts: {', '.join(scripts) or 'none'})")

    def close(self):
        self.closed = True
        self.live.set(False)
        if self.webcam:
            self.webcam.stop()
        if self.ai:
            self.ai.stop()
        # cancel pending after() callbacks so Tk doesn't report them after destroy
        for after_id in self.root.tk.call("after", "info"):
            self.root.after_cancel(after_id)
        self.root.destroy()


def main():
    try:  # crisp (not bitmap-stretched) rendering on high-DPI Windows displays
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = TkinterDnD.Tk() if HAVE_DND else tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
