# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless"]
# ///
"""
Convert an image into a line drawing and write URScript programs that
draw it relative to the PolyScope feature "Feature1".

Styles:
    edges  Canny edge outlines only (default)
    comic  Comic look: smoothed + posterised image, XDoG ink lines,
           and shading drawn as hatching / cross-hatching

Colours:
    The robot can draw with up to 4 pens (PEN_COLOURS). Each stroke gets the pen
    colour that best matches the source image underneath it, and one script is
    written per colour, e.g. monalisa_red.script, monalisa_blue.script.

Usage:
    uv run src/image_to_urscript.py images/monalisa.png
    uv run src/image_to_urscript.py images/monalisa.png --style comic
    uv run src/image_to_urscript.py images/monalisa.png --style comic --colours black,red,green,blue
    uv run src/image_to_urscript.py photo.jpg -o photo.script --width 120

Outputs:
    <name>.script            URScript program (single pen, no --colours)
    <name>_<colour>.script   one program per pen colour (with --colours)
    <name>_preview.png       Preview of the strokes in pen colours
    <name>_comic.png         (comic style only) intermediate posterised + inked image
    <name>_colours.png       (with --colours) which pen colour covers which area
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Properties – adjust these to suit the canvas, pen and image
# ---------------------------------------------------------------------------

# Drawing size on the canvas. The image is scaled to IMAGE_WIDTH_MM wide;
# height follows the aspect ratio, but is limited to IMAGE_MAX_HEIGHT_MM.
IMAGE_WIDTH_MM = 150.0
IMAGE_MAX_HEIGHT_MM = 200.0

# Offset of the drawing's lower-left corner from the feature origin (mm)
OFFSET_X_MM = 0.0
OFFSET_Y_MM = 0.0

# Pen heights along the feature Z axis (mm). Pen touches canvas at PEN_DOWN_Z.
# If the feature's Z axis points INTO the canvas, make PEN_UP_Z_MM negative.
PEN_DOWN_Z_MM = 0.0
PEN_UP_Z_MM = 10.0

# Mirror options, in case the feature's axes don't match the image view
FLIP_X = False
FLIP_Y = False

# Tool orientation (rotation vector, same as the original square program)
ORIENT_RX = 1.6
ORIENT_RY = -1.4
ORIENT_RZ = -0.8

# Motion (m/s, m/s^2)
DRAW_SPEED = 0.05
TRAVEL_SPEED = 0.10
ACCELERATION = 0.2
BLEND_RADIUS_MM = 0.5  # smoothing within a stroke; 0 = stop at every point

# Line extraction (both styles)
WORK_RESOLUTION_PX = 500     # longest image side is resized to this before processing
SIMPLIFY_TOLERANCE_MM = 0.3  # max deviation when reducing points along a line
MIN_LINE_LENGTH_MM = 2.0     # strokes shorter than this are dropped

# --- Pen colours ---
# The pens available on the robot, as BGR values. Used to match image colours
# and to draw the preview. Order = drawing order in the preview.
PEN_COLOURS = {
    "black": (0, 0, 0),
    "green": (40, 160, 0),
    "red": (30, 30, 220),
    "blue": (220, 60, 0),
}
ACTIVE_COLOURS = ["black"]   # pens in use; strokes are only assigned to these

# How image colours are matched to pens:
#   "hue"     – colourful pixels go to the pen with the nearest hue; grey / dark pixels go
#               to black (if active). Good for paintings and photos.
#   "nearest" – nearest pen colour in Lab colour space. Good for flat, graphic images.
COLOUR_METHOD = "hue"
COLOUR_MIN_SATURATION = 70   # (hue method) 0-255; less saturated pixels count as black
COLOUR_DARK_VALUE = 60       # (hue method) 0-255; darker pixels count as black
COLOUR_PAPER_VALUE = 200     # light, unsaturated pixels are "paper": they take the colour
                             # of the nearest non-paper area (so outlines get the object's colour)
COLOUR_SMOOTH_PX = 7         # median filter on the colour map, removes colour speckle
COLOUR_MIN_RUN_PX = 6        # a line only switches colour after this many pixels

# --- "edges" style ---
BLUR_KERNEL = 5   # Gaussian blur (odd number), removes noise / fine texture
CANNY_LOW = 50    # Canny thresholds: raise these for fewer, stronger lines
CANNY_HIGH = 150

# --- "comic" style ---
COMIC_INK = True    # draw ink lines
COMIC_HATCH = True  # draw hatching

# 0. Contrast: histogram equalisation spreads dark images (like the Mona Lisa)
#    across all tones so hatching doesn't cover everything
EQUALIZE = True

# 1. Flatten: repeated edge-preserving bilateral filter
SMOOTH_ITERATIONS = 4
SMOOTH_DIAMETER = 9
SMOOTH_SIGMA_COLOR = 40
SMOOTH_SIGMA_SPACE = 9

# 2. Posterise into flat tones (k-means on brightness)
TONE_LEVELS = 4

# 3. Ink lines: edge term of XDoG (difference of Gaussians), thresholded
INK_SMOOTH_ITERATIONS = 1  # ink uses lighter smoothing than the tones, keeps facial detail
INK_SIGMA = 1.0         # line scale in px; larger = bolder, fewer lines
INK_K = 1.6             # ratio between the two Gaussians
INK_THRESHOLD = 0.010   # lower = more lines
INK_MIN_AREA_PX = 30    # drop ink specks smaller than this before thinning

# 4. Hatching: (angle in degrees, line spacing in mm, darkest..max_tone)
#    Each layer hatches every region whose tone index (0 = darkest) is <= max_tone,
#    so dark regions accumulate layers -> cross-hatching.
HATCH_LAYERS = [
    (45, 3.0, 1),
    (-45, 3.0, 0),
]
HATCH_MIN_REGION_PX = 40  # ignore tone regions smaller than this

# ---------------------------------------------------------------------------

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def mm_per_pixel(shape) -> float:
    h_px, w_px = shape[:2]
    scale = IMAGE_WIDTH_MM / w_px
    if h_px * scale > IMAGE_MAX_HEIGHT_MM:
        scale = IMAGE_MAX_HEIGHT_MM / h_px
    return scale


def read_image(image_path: Path) -> np.ndarray:
    """Read an image as BGR. np.fromfile + imdecode copes with non-ASCII Windows paths."""
    data = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img


def save_image(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    buf.tofile(str(path))


def prepare(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """BGR (or gray) image -> (gray, bgr), resized so the longest side is WORK_RESOLUTION_PX."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = WORK_RESOLUTION_PX / max(h, w)
    bgr = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr


def active_colours() -> list[str]:
    cols = [c for c in PEN_COLOURS if c in ACTIVE_COLOURS]
    return cols or ["black"]


# --------------------------- colour assignment ------------------------------

def colour_labels(bgr: np.ndarray) -> np.ndarray:
    """Per-pixel index into active_colours(): which pen should draw here."""
    cols = active_colours()
    h, w = bgr.shape[:2]
    if len(cols) == 1:
        return np.zeros((h, w), np.uint8)
    img = cv2.medianBlur(bgr, 5)
    pens = np.array([PEN_COLOURS[c] for c in cols], np.uint8).reshape(1, -1, 3)

    if COLOUR_METHOD == "nearest":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).reshape(-1, 1, 3).astype(np.float32)
        pen_lab = cv2.cvtColor(pens, cv2.COLOR_BGR2Lab).reshape(1, -1, 3).astype(np.float32)
        # down-weight lightness: otherwise light tints (e.g. pale blue) match the lightest
        # pen instead of the pen with the same colour
        weight = np.array([0.3, 1.0, 1.0], np.float32)
        labels = np.linalg.norm((lab - pen_lab) * weight, axis=2).argmin(axis=1).reshape(h, w)
    else:  # "hue"
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        chroma = [i for i, c in enumerate(cols) if c != "black"]
        black = cols.index("black") if "black" in cols else None
        if chroma:
            pen_hue = cv2.cvtColor(pens, cv2.COLOR_BGR2HSV)[0, chroma, 0].astype(np.int32)
            d = np.abs(hue[..., None] - pen_hue[None, None, :])
            d = np.minimum(d, 180 - d)                      # hue is circular (0..179)
            labels = np.array(chroma)[d.argmin(axis=2)]
        else:
            labels = np.full((h, w), black)
        if black is not None:
            grey_or_dark = (sat < COLOUR_MIN_SATURATION) | (val < COLOUR_DARK_VALUE)
            labels = np.where(grey_or_dark, black, labels)

    labels = labels.astype(np.uint8)

    # Paper (white/light grey) has no pen colour of its own: give it the colour of the
    # nearest non-paper pixel, so an outline between a red object and white paper is red.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    paper = (hsv[..., 2] >= COLOUR_PAPER_VALUE) & (hsv[..., 1] < COLOUR_MIN_SATURATION)
    if paper.any() and not paper.all():
        _, nearest = cv2.distanceTransformWithLabels(
            paper.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        # each non-paper pixel carries its own label id: map label id -> its pen index
        lut = np.zeros(int(nearest.max()) + 1, np.uint8)
        lut[nearest[~paper]] = labels[~paper]
        labels = lut[nearest]

    k = int(COLOUR_SMOOTH_PX) | 1
    if k > 1:
        labels = cv2.medianBlur(labels, min(k, 255))
    return labels


def colour_map_image(labels: np.ndarray) -> np.ndarray:
    """Render the label map in pen colours (shows which pen covers which area)."""
    palette = np.array([PEN_COLOURS[c] for c in active_colours()], np.uint8)
    img = palette[labels]
    # lighten so black areas and lines remain distinguishable in a thumbnail
    return cv2.addWeighted(img, 0.6, np.full_like(img, 255), 0.4, 0)


def split_by_colour(stroke: np.ndarray, labels: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Cut a pixel polyline (row, col) wherever the pen colour underneath changes.
    Short colour runs are merged into the previous run to avoid confetti."""
    rows = np.clip(np.round(stroke[:, 0]).astype(int), 0, labels.shape[0] - 1)
    cols = np.clip(np.round(stroke[:, 1]).astype(int), 0, labels.shape[1] - 1)
    lab = labels[rows, cols]
    if (lab == lab[0]).all():
        return [(int(lab[0]), stroke)]
    # run-length encode, then absorb short runs
    starts = np.r_[0, np.nonzero(np.diff(lab))[0] + 1]
    runs = [[int(lab[s]), s, e] for s, e in zip(starts, np.r_[starts[1:], len(lab)])]
    merged = []
    for run in runs:
        if merged and (run[2] - run[1] < COLOUR_MIN_RUN_PX or merged[-1][0] == run[0]):
            merged[-1][2] = run[2]
        else:
            merged.append(run)
    if len(merged) > 1 and merged[0][2] - merged[0][1] < COLOUR_MIN_RUN_PX:
        merged[1][1] = 0
        merged.pop(0)
    # neighbouring pieces share their boundary point so the line stays continuous
    return [(c, stroke[max(s - 1, 0):e]) for c, s, e in merged]


# --------------------------- "edges" style ---------------------------------

def edge_strokes(gray: np.ndarray) -> list[np.ndarray]:
    k = int(BLUR_KERNEL) | 1  # kernel must be odd
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH) > 0
    return trace_strokes(edges)


# --------------------------- "comic" style ---------------------------------

def smooth(gray: np.ndarray, iterations: int) -> np.ndarray:
    out = gray
    for _ in range(iterations):
        out = cv2.bilateralFilter(out, SMOOTH_DIAMETER, SMOOTH_SIGMA_COLOR, SMOOTH_SIGMA_SPACE)
    return out


def posterise(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (tone index per pixel with 0 = darkest, tone brightness values)."""
    cv2.setRNGSeed(0)
    data = gray.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(data, TONE_LEVELS, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    order = np.argsort(centers.ravel())         # dark -> light
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    tones = rank[labels.ravel()].reshape(gray.shape)
    return tones, centers.ravel()[order]


def ink_lines(gray: np.ndarray) -> np.ndarray:
    """XDoG edge term: dark side of edges where the DoG response exceeds a threshold."""
    img = gray.astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(img, (0, 0), INK_SIGMA)
    g2 = cv2.GaussianBlur(img, (0, 0), INK_SIGMA * INK_K)
    ink = ((g2 - g1) > INK_THRESHOLD).astype(np.uint8)
    # remove specks
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= INK_MIN_AREA_PX
    return keep[labels]


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning: reduce thick ink lines to 1-px centre lines (vectorised numpy)."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            # neighbours P2..P9, clockwise from north
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            count = sum(n.astype(np.int32) for n in nb)
            trans = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(np.int32) for i in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (img[1:-1, 1:-1] == 1) & (count >= 2) & (count <= 6) & (trans == 1) & cond
            if remove.any():
                img[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def hatch_strokes(tones: np.ndarray, px_per_mm: float,
                  labels: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Hatch lines clipped to tone regions; each line is also cut where the pen colour changes."""
    h, w = tones.shape
    kernel = np.ones((3, 3), np.uint8)
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)  # (x, y)
    strokes = []
    for angle_deg, spacing_mm, max_tone in HATCH_LAYERS:
        mask = (tones <= max_tone).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        n, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= HATCH_MIN_REGION_PX
        mask = keep[comp]

        a = np.deg2rad(angle_deg)
        d = np.array([np.cos(a), np.sin(a)], np.float32)   # along the hatch line
        nrm = np.array([-d[1], d[0]], np.float32)            # across the lines
        t_min, t_max = (corners @ d).min(), (corners @ d).max()
        o_min, o_max = (corners @ nrm).min(), (corners @ nrm).max()
        t = np.arange(t_min, t_max, 0.5, dtype=np.float32)
        spacing_px = max(spacing_mm, 0.3) * px_per_mm

        for o in np.arange(o_min + spacing_px / 2, o_max, spacing_px):
            x = nrm[0] * o + d[0] * t
            y = nrm[1] * o + d[1] * t
            xi, yi = np.round(x).astype(int), np.round(y).astype(int)
            inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            key = np.full(len(t), -1, np.int32)   # -1 = no hatching, else pen index
            on = np.zeros(len(t), bool)
            on[inside] = mask[yi[inside], xi[inside]]
            key[on] = labels[yi[on], xi[on]]
            # split into runs of constant key
            change = np.nonzero(np.diff(key))[0] + 1
            for s, e in zip(np.r_[0, change], np.r_[change, len(t)]):
                if key[s] >= 0 and e - 1 > s:
                    strokes.append((int(key[s]),
                                    np.array([[y[s], x[s]], [y[e - 1], x[e - 1]]], np.float32)))
    return strokes


def comic_strokes(gray: np.ndarray, labels: np.ndarray):
    """Return (coloured pixel strokes, intermediate posterised + inked image)."""
    if EQUALIZE:
        gray = cv2.equalizeHist(gray)
    tones, levels = posterise(smooth(gray, SMOOTH_ITERATIONS))
    comic = levels[tones].astype(np.uint8)
    strokes = []
    if COMIC_INK:
        ink = ink_lines(smooth(gray, INK_SMOOTH_ITERATIONS))
        comic[ink] = 0
        for s in trace_strokes(thin(ink)):
            strokes += split_by_colour(s, labels)
    if COMIC_HATCH:
        strokes += hatch_strokes(tones, 1.0 / mm_per_pixel(gray.shape), labels)
    return strokes, comic


# --------------------------- shared ----------------------------------------

def trace_strokes(edges: np.ndarray) -> list[np.ndarray]:
    """Walk a 1-pixel-wide line map into open polylines (row, col)."""
    h, w = edges.shape
    remaining = edges.copy()

    def neighbours(r, c):
        for dr, dc in NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and remaining[rr, cc]:
                yield rr, cc

    def walk(r, c):
        path = [(r, c)]
        remaining[r, c] = False
        while True:
            nxt = next(neighbours(r, c), None)
            if nxt is None:
                return path
            r, c = nxt
            remaining[r, c] = False
            path.append((r, c))

    strokes = []
    # Start at line ends first (pixels with one neighbour) so lines aren't split in the middle
    pixels = list(zip(*np.nonzero(edges)))
    endpoints = [p for p in pixels if sum(1 for _ in neighbours(*p)) == 1]
    for r, c in endpoints + pixels:
        if not remaining[r, c]:
            continue
        path = walk(r, c)
        # Extend backwards from the start point too, in case we began mid-line
        tail = []
        nxt = next(neighbours(*path[0]), None)
        if nxt is not None:
            tail = walk(*nxt)
        strokes.append(np.array(tail[::-1] + path, dtype=np.float32))
    return strokes


def stroke_to_mm(s: np.ndarray, shape) -> np.ndarray | None:
    """Pixel (row, col) polyline -> simplified mm polyline in the feature frame, or None if too short."""
    h_px, w_px = shape[:2]
    mm_per_px = mm_per_pixel(shape)
    width_mm, height_mm = w_px * mm_per_px, h_px * mm_per_px
    x = s[:, 1] * mm_per_px
    y = (h_px - 1 - s[:, 0]) * mm_per_px  # image rows go down, feature Y goes up
    if FLIP_X:
        x = width_mm - x
    if FLIP_Y:
        y = height_mm - y
    pts = np.stack([x + OFFSET_X_MM, y + OFFSET_Y_MM], axis=1).astype(np.float32)
    pts = cv2.approxPolyDP(pts.reshape(-1, 1, 2), SIMPLIFY_TOLERANCE_MM, False).reshape(-1, 2)
    if len(pts) < 2 or np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() < MIN_LINE_LENGTH_MM:
        return None
    return pts


def order_strokes(strokes):
    """Greedy nearest-neighbour ordering (strokes may be reversed) to cut pen-up travel."""
    if not strokes:
        return strokes
    todo = list(strokes)
    starts = np.array([s[0] for s in todo])
    ends = np.array([s[-1] for s in todo])
    used = np.zeros(len(todo), bool)
    ordered = []
    pos = np.array([OFFSET_X_MM, OFFSET_Y_MM])
    for _ in range(len(todo)):
        d_start = np.linalg.norm(starts - pos, axis=1)
        d_end = np.linalg.norm(ends - pos, axis=1)
        d_start[used] = np.inf
        d_end[used] = np.inf
        i_s, i_e = int(d_start.argmin()), int(d_end.argmin())
        if d_start[i_s] <= d_end[i_e]:
            s, i = todo[i_s], i_s
        else:
            s, i = todo[i_e][::-1], i_e
        used[i] = True
        ordered.append(s)
        pos = s[-1]
    return ordered


def estimate_time_s(strokes):
    draw = sum(np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in strokes)
    travel = sum(np.linalg.norm(b[0] - a[-1]) for a, b in zip(strokes, strokes[1:]))
    lift = 2 * abs(PEN_UP_Z_MM - PEN_DOWN_Z_MM) * len(strokes)
    # Rough: ignores acceleration, which dominates with many short segments
    return draw / (DRAW_SPEED * 1000) + travel / (TRAVEL_SPEED * 1000) + lift / (DRAW_SPEED * 1000)


@dataclass
class Result:
    style: str
    layers: dict              # pen colour -> ordered strokes in mm (feature frame)
    width_mm: float
    height_mm: float
    comic: np.ndarray | None = None       # intermediate comic image (comic style only)
    colour_map: np.ndarray | None = None  # pen colour per area (only with >1 colour)
    est_times: dict = field(default_factory=dict)  # pen colour -> seconds

    @property
    def strokes(self) -> list:
        return [s for strokes in self.layers.values() for s in strokes]

    @property
    def n_points(self) -> int:
        return sum(len(s) for s in self.strokes)

    @property
    def draw_length_mm(self) -> float:
        return float(sum(np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in self.strokes))

    @property
    def est_time_s(self) -> float:
        return sum(self.est_times.values())


def process(img: np.ndarray, style: str) -> Result:
    """Run the full pipeline on a BGR image using the current module settings."""
    gray, bgr = prepare(img)
    cols = active_colours()
    labels = colour_labels(bgr)

    comic = None
    if style == "comic":
        pixel_strokes, comic = comic_strokes(gray, labels)
    else:
        pixel_strokes = [p for s in edge_strokes(gray) for p in split_by_colour(s, labels)]

    layers = {c: [] for c in cols}
    for label, s in pixel_strokes:
        pts = stroke_to_mm(s, gray.shape)
        if pts is not None:
            layers[cols[label]].append(pts)
    layers = {c: order_strokes(strokes) for c, strokes in layers.items()}

    mm_px = mm_per_pixel(gray.shape)
    return Result(
        style=style, layers=layers,
        width_mm=gray.shape[1] * mm_px, height_mm=gray.shape[0] * mm_px,
        comic=comic,
        colour_map=colour_map_image(labels) if len(cols) > 1 else None,
        est_times={c: estimate_time_s(s) for c, s in layers.items()},
    )


# --------------------------- output ----------------------------------------

def blend_radii(pts):
    """Blend radius per point, kept below half the adjacent segment lengths (UR requirement)."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    radii = np.zeros(len(pts))
    for i in range(1, len(pts) - 1):
        radii[i] = min(BLEND_RADIUS_MM, 0.45 * seg[i - 1], 0.45 * seg[i])
    return radii  # first/last points: 0 (stop exactly at stroke ends)


def m(v_mm):
    return f"{v_mm / 1000.0:.5f}"


def urscript_text(result: Result, source_name: str, colour: str) -> str:
    """URScript program drawing only the strokes of one pen colour."""
    strokes = result.layers.get(colour, [])
    n_points = sum(len(s) for s in strokes)
    lines = [
        f"# Line drawing ({result.style} style) of '{source_name}' relative to feature \"Feature1\"",
        f"# PEN COLOUR: {colour.upper()}  - fit the {colour} pen before running",
        f"# Size: {result.width_mm:.1f} mm x {result.height_mm:.1f} mm, "
        f"{len(strokes)} strokes, {n_points} points",
        "# Generated by image_to_urscript.py",
        "",
        "myFeature = Feature1",
        f'textmsg("Pen colour: {colour}")',
        'textmsg("Feature Pose:", myFeature)',
        "start_pos = get_actual_tcp_pose()",
        'textmsg("Start Pos.", start_pos)',
        "start_local = pose_trans(pose_inv(myFeature), start_pos)",
        'textmsg("Local TCP:", start_local)',
        "",
        f"draw_v = {DRAW_SPEED}",
        f"travel_v = {TRAVEL_SPEED}",
        f"acc = {ACCELERATION}",
        f"z_down = {m(PEN_DOWN_Z_MM)}",
        f"z_up = {m(PEN_UP_Z_MM)}",
        f"rx = {ORIENT_RX}",
        f"ry = {ORIENT_RY}",
        f"rz = {ORIENT_RZ}",
        "",
        "# Lift pen, travel to (x, y), lower pen",
        "def pen_to(x, y):",
        "  movel(pose_trans(myFeature, p[x, y, z_up, rx, ry, rz]), a=acc, v=travel_v)",
        "  movel(pose_trans(myFeature, p[x, y, z_down, rx, ry, rz]), a=acc, v=draw_v)",
        "end",
        "",
        "# Draw a line to (x, y) with blend radius r",
        "def line_to(x, y, r):",
        "  movel(pose_trans(myFeature, p[x, y, z_down, rx, ry, rz]), a=acc, v=draw_v, r=r)",
        "end",
        "",
        "def pen_up():",
        "  pose = get_actual_tcp_pose()",
        "  local = pose_trans(pose_inv(myFeature), pose)",
        "  movel(pose_trans(myFeature, p[local[0], local[1], z_up, rx, ry, rz]), a=acc, v=travel_v)",
        "end",
        "",
    ]
    for i, s in enumerate(strokes, 1):
        radii = blend_radii(s)
        lines.append(f"# stroke {i}/{len(strokes)}")
        lines.append(f"pen_to({m(s[0][0])}, {m(s[0][1])})")
        for (x, y), r in zip(s[1:], radii[1:]):
            lines.append(f"line_to({m(x)}, {m(y)}, {m(r)})")
        lines.append("pen_up()")
    lines.append(f'textmsg("Drawing finished ({colour})")')
    lines.append("# End")
    return "\n".join(lines) + "\n"


def render_preview(result: Result, px_per_mm: float = 5) -> np.ndarray:
    """Draw the strokes in their pen colours as a BGR image (feature +Y is up). Render at
    the display size rather than downscaling a large render, which aliases hatching badly."""
    margin = 5
    w = int((result.width_mm + abs(OFFSET_X_MM)) * px_per_mm) + 2 * margin
    h = int((result.height_mm + abs(OFFSET_Y_MM)) * px_per_mm) + 2 * margin
    canvas = np.full((h, w, 3), 255, np.uint8)
    for colour, strokes in result.layers.items():
        bgr = tuple(int(v) for v in PEN_COLOURS.get(colour, (0, 0, 0)))
        for s in strokes:
            pts = np.column_stack([s[:, 0] * px_per_mm + margin, h - margin - s[:, 1] * px_per_mm])
            cv2.polylines(canvas, [pts.round().astype(np.int32)], False, bgr, 1, cv2.LINE_AA)
    return canvas


def main():
    global IMAGE_WIDTH_MM, ACTIVE_COLOURS
    parser = argparse.ArgumentParser(description="Convert an image to a URScript line drawing.")
    parser.add_argument("image", type=Path, help="input image file, e.g. monalisa.png")
    parser.add_argument("-o", "--output", type=Path, help="output .script (default: <image>.script)")
    parser.add_argument("--width", type=float, help=f"drawing width in mm (default {IMAGE_WIDTH_MM})")
    parser.add_argument("--style", choices=["edges", "comic"], default="edges",
                        help="edges = outlines only, comic = ink lines + hatching (default edges)")
    parser.add_argument("--colours", "--colors", dest="colours",
                        help=f"comma-separated pens to use, from: {','.join(PEN_COLOURS)}. "
                             "Writes one <name>_<colour>.script per pen.")
    args = parser.parse_args()

    if args.width:
        IMAGE_WIDTH_MM = args.width
    if args.colours:
        wanted = [c.strip().lower() for c in args.colours.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in PEN_COLOURS]
        if unknown:
            raise SystemExit(f"Unknown colour(s): {', '.join(unknown)}. Available: {', '.join(PEN_COLOURS)}")
        ACTIVE_COLOURS = wanted
    out_path = args.output or args.image.with_suffix(".script")
    stem = out_path.stem

    try:
        img = read_image(args.image)
    except ValueError as e:
        raise SystemExit(str(e))
    result = process(img, args.style)

    written = []
    for colour, strokes in result.layers.items():
        if args.colours:
            if not strokes:
                continue
            path = out_path.with_name(f"{stem}_{colour}.script")
        else:
            path = out_path
        path.write_text(urscript_text(result, args.image.name, colour), encoding="utf-8")
        written.append((colour, path, len(strokes), result.est_times[colour]))

    preview_path = out_path.with_name(stem + "_preview.png")
    save_image(preview_path, render_preview(result))
    extra = []
    if result.comic is not None:
        extra.append(("Comic image", out_path.with_name(stem + "_comic.png"), result.comic))
    if result.colour_map is not None:
        extra.append(("Colour map", out_path.with_name(stem + "_colours.png"), result.colour_map))
    for _, path, img_out in extra:
        save_image(path, img_out)

    print(f"Style        : {result.style}")
    print(f"Drawing size : {result.width_mm:.1f} x {result.height_mm:.1f} mm")
    print(f"Strokes      : {len(result.strokes)}  ({result.n_points} points)")
    print(f"Est. time    : ~{result.est_time_s / 60:.0f} min (lower bound, excl. pen changes)")
    for colour, path, n, t in written:
        print(f"  {colour:<6} : {n:4d} strokes  ~{t / 60:4.1f} min  -> {path}")
    print(f"Preview      : {preview_path}")
    for label, path, _ in extra:
        print(f"{label:<13}: {path}")


if __name__ == "__main__":
    main()
