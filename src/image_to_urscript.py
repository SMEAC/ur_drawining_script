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
           and shading (zig-zag colouring in, or hatch lines)
    lineart  The input already IS a line drawing (e.g. from the AI restyle):
             its dark lines are traced along their centres, one pen stroke per line

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
import random
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

# --- "lineart" style (input is already a line drawing) ---
LINEART_RESOLUTION_PX = 1200  # work resolution; higher keeps fine, close-together lines apart
LINEART_THRESHOLD = 225       # 0-255: pixels darker than this are ink (raise for faint lines)
LINEART_BRIDGE_PX = 0         # close breaks in faint lines up to ~2x this; 0 = off
LINEART_MIN_AREA_PX = 15      # drop ink specks smaller than this
LINEART_SPUR_PX = 6           # prune side branches of the centre line shorter than this
LINEART_SMOOTH_PX = 2         # smooth the traced centre line over +-this many pixels
LINEART_SIMPLIFY_MM = 0.1     # point reduction tolerance (finer than SIMPLIFY_TOLERANCE_MM)
LINEART_MIN_LENGTH_MM = 0.8   # shorter strokes are dropped (lashes and teeth are short)
LINEART_FILL_THICK = True     # fill ink wider than ~2 pen widths (pupils, bold lashes)
                              # with nested outlines, instead of only its centre line
LINEART_PEN_MM = 0.5          # pen line width, the spacing of those nested outlines

# --- "comic" style ---
COMIC_INK = True    # draw ink lines
COMIC_HATCH = True  # draw shading fill

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

# 4. Shading fill: (angle in degrees, line spacing in mm, darkest..max_tone)
#    Each layer fills every region whose tone index (0 = darkest) is <= max_tone,
#    so dark regions get several passes -> cross-hatching.
HATCH_LAYERS = [
    (45, 3.0, 1),
    (-45, 3.0, 0),
]
HATCH_MIN_REGION_PX = 40  # ignore tone regions smaller than this

# How shading regions are filled:
#   "zigzag"     – V V V: continuous diagonals from edge to opposite edge, like scribbling
#   "z-lines"    – Z Z Z: lines all drawn the same way, each followed by a diagonal back
#   "serpentine" – back-and-forth lines joined by short (optionally rounded) turns
#   "lines"      – separate straight hatch lines (pen lifts after every line)
FILL_STYLE = "zigzag"
FILL_STYLES = ["zigzag", "z-lines", "serpentine", "lines"]
# "shape" – fill along each region's long axis (layer angles become relative:
#           the first layer follows the shape, the others keep their angle difference)
# "fixed" – use the layer angles as given
FILL_DIRECTION = "shape"
FILL_SHAPE_MIN_ELONGATION = 2.0  # ("shape") only regions at least this elongated follow their
                                 # own axis; rounder regions keep the layer angle
FILL_ROUND_TURNS = True  # (serpentine) loop-shaped turns instead of sharp corners
FILL_JITTER_MM = 0.3      # random variation of spacing and line ends; 0 = machine-perfect
FILL_SEED = 0             # jitter is repeatable for the same seed

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


def prepare(img: np.ndarray, resolution: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """BGR (or gray) image -> (gray, bgr), resized so the longest side is `resolution`
    (default WORK_RESOLUTION_PX)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = (resolution or WORK_RESOLUTION_PX) / max(h, w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    bgr = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=interp)
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


# --------------------------- "lineart" style -------------------------------

def neighbour_count(mask: np.ndarray) -> np.ndarray:
    """Number of 8-connected set neighbours of every pixel."""
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    return cv2.filter2D(mask.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT).round().astype(np.int32)


def remove_staircases(skel: np.ndarray) -> np.ndarray:
    """Delete the corner pixel of every L-shaped step (e.g. set N and E neighbours, empty S, SW, W).
    Thinning leaves these on diagonal lines; they give pixels 3 neighbours, which would look
    like junctions and cut the line. The two neighbours stay diagonally connected."""
    img = np.pad(skel, 1)
    for a, b, opp in (((-1, 0), (0, 1), ((1, 0), (1, -1), (0, -1))),    # N, E
                      ((0, 1), (1, 0), ((0, -1), (-1, -1), (-1, 0))),   # E, S
                      ((1, 0), (0, -1), ((-1, 0), (-1, 1), (0, 1))),    # S, W
                      ((0, -1), (-1, 0), ((0, 1), (1, 1), (1, 0)))):    # W, N
        def at(d):
            return img[1 + d[0]:img.shape[0] - 1 + d[0], 1 + d[1]:img.shape[1] - 1 + d[1]]
        corner = img[1:-1, 1:-1] & at(a) & at(b)
        for d in opp:
            corner &= ~at(d)
        img[1:-1, 1:-1] &= ~corner
    return img[1:-1, 1:-1]


def prune_spurs(skel: np.ndarray, length: int) -> np.ndarray:
    """Remove short side branches ("spurs") that thinning leaves on thick or blobby lines.
    Classic pruning: strip line ends `length` times, then regrow only from the ends that
    survived, so real lines get their full length back but spurs at junctions don't."""
    if length <= 0:
        return skel
    pruned = skel.copy()
    for _ in range(length):
        ends = pruned & (neighbour_count(pruned) <= 1)
        if not ends.any():
            break
        pruned &= ~ends
    grow = pruned & (neighbour_count(pruned) == 1)
    kernel = np.ones((3, 3), np.uint8)
    regrown = grow
    for _ in range(length):
        regrown = (cv2.dilate(regrown.astype(np.uint8), kernel) > 0) & skel & ~pruned
        regrown |= grow
    return pruned | regrown


def lineart_mask(gray: np.ndarray) -> np.ndarray:
    """Dark ink pixels of a line drawing, without specks."""
    ink = (cv2.GaussianBlur(gray, (3, 3), 0) < LINEART_THRESHOLD).astype(np.uint8)
    if LINEART_BRIDGE_PX > 0:  # close small breaks in faint, grainy lines
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * LINEART_BRIDGE_PX + 1,) * 2)
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= LINEART_MIN_AREA_PX
    return keep[labels]


def order_chain(pixels: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], bool]:
    """Order the pixels of a junction-free skeleton piece into a path. Returns (path, closed)."""
    todo = set(pixels)

    def nbrs(p):
        return [(p[0] + dr, p[1] + dc) for dr, dc in NEIGHBOURS if (p[0] + dr, p[1] + dc) in todo]

    ends = [p for p in pixels if len(nbrs(p)) <= 1]
    start = ends[0] if ends else pixels[0]
    path = [start]
    todo.discard(start)
    while True:
        nxt = nbrs(path[-1])
        if not nxt:
            break
        # prefer edge neighbours over diagonals so staircase pixels aren't skipped
        nxt.sort(key=lambda q: abs(q[0] - path[-1][0]) + abs(q[1] - path[-1][1]))
        path.append(nxt[0])
        todo.discard(nxt[0])
    return path, not ends and len(path) > 2


def skeleton_strokes(skel: np.ndarray) -> list[np.ndarray]:
    """Turn a 1-px skeleton into long strokes. The skeleton is cut at junctions into chains;
    at every junction the chain ends are paired up by how straight they continue, so a line
    that crosses or touches another runs on as one stroke instead of breaking into pieces."""
    junction = skel & (neighbour_count(skel) >= 3)
    n_nodes, node_lab = cv2.connectedComponents(junction.astype(np.uint8), connectivity=8)
    chain_mask = (skel & ~junction).astype(np.uint8)
    n_chains, chain_lab = cv2.connectedComponents(chain_mask, connectivity=8)

    rows, cols = np.nonzero(chain_mask)
    labs = chain_lab[rows, cols]
    srt = np.argsort(labs, kind="stable")
    groups = np.split(np.stack([rows[srt], cols[srt]], 1), np.flatnonzero(np.diff(labs[srt])) + 1)
    nr, nc_ = np.nonzero(junction)
    node_lab_px = node_lab[nr, nc_]
    node_centre = np.zeros((n_nodes, 2))
    np.add.at(node_centre, node_lab_px, np.stack([nr, nc_], 1))
    node_centre /= np.maximum(np.bincount(node_lab_px, minlength=n_nodes), 1)[:, None]
    h, w = skel.shape

    chains, closed = [], []
    node_ends = {}  # node -> [(chain index, end 0/1, outward direction)]
    for g in groups:
        if len(g) == 0:
            continue
        path, is_closed = order_chain([tuple(p) for p in g])
        ci = len(chains)
        chains.append(np.array(path, np.float32))
        closed.append(is_closed)
        if is_closed:
            continue
        touching = []
        for p in (path[0], path[-1]):
            touching.append(sorted({node_lab[r, c] for r, c in
                                    ((p[0] + dr, p[1] + dc) for dr, dc in NEIGHBOURS)
                                    if 0 <= r < h and 0 <= c < w and node_lab[r, c] > 0}))
        if len(path) == 1:
            # a 1-px piece between two junctions: one end faces each of them
            touching = [touching[0][:1], touching[0][1:]]
        for end, (p, inner) in enumerate(((path[0], path[min(6, len(path) - 1)]),
                                          (path[-1], path[max(-7, -len(path))]))):
            d = np.subtract(p, inner, dtype=np.float64)
            if not d.any():  # 1-px piece: direction towards its junction
                d = node_centre[touching[end][0]] - p if touching[end] else d
            for node in touching[end]:
                node_ends.setdefault(node, []).append((ci, end, d / (np.linalg.norm(d) or 1)))

    # pair chain ends at each junction, straightest continuation first
    link = {}   # (chain, end) -> (other chain, other end, node)
    tip = {}    # (chain, end) -> node the unpaired end touches (extend the line to it)
    for node, ends in node_ends.items():
        cands = []
        for i in range(len(ends)):
            for j in range(i + 1, len(ends)):
                (ci, ei, di), (cj, ej, dj) = ends[i], ends[j]
                if ci != cj:
                    cands.append((float(-di @ dj), i, j))
        cands.sort(reverse=True)
        used = set()
        for straightness, i, j in cands:
            # a junction of exactly two ends is just a bend: always join it
            if i in used or j in used or (straightness < 0.5 and len(ends) > 2):
                continue
            used |= {i, j}
            (ci, ei, _), (cj, ej, _) = ends[i], ends[j]
            if (ci, ei) not in link and (cj, ej) not in link:
                link[(ci, ei)] = (cj, ej, node)
                link[(cj, ej)] = (ci, ei, node)
        for k, (ci, ei, _) in enumerate(ends):
            if k not in used:
                tip.setdefault((ci, ei), node)

    strokes, done = [], set()

    def follow(ci, entry_end):
        """Walk linked chains starting with chain ci entered at entry_end."""
        pts = []
        while ci not in done:
            done.add(ci)
            seg = chains[ci] if entry_end == 0 else chains[ci][::-1]
            pts.append(seg)
            exit_end = 1 - entry_end
            nxt = link.get((ci, exit_end))
            if nxt is None:
                if (ci, exit_end) in tip:
                    pts.append(node_centre[tip[(ci, exit_end)]][None].astype(np.float32))
                break
            cj, ej, node = nxt
            pts.append(node_centre[node][None].astype(np.float32))
            ci, entry_end = cj, ej
        return pts

    # start at chains with a free end, so each stroke runs from one free end to another
    order = [(ci, e) for ci in range(len(chains)) if not closed[ci] for e in (0, 1)
             if (ci, e) not in link] + [(ci, 0) for ci in range(len(chains))]
    for ci, e in order:
        if ci in done:
            continue
        head = []
        if (ci, e) in tip:
            head = [node_centre[tip[(ci, e)]][None].astype(np.float32)]
        pts = follow(ci, e)
        s = np.concatenate(head + pts)
        if closed[ci] and len(pts) == 1:
            s = np.concatenate([s, s[:1]])
        strokes.append(s)
    return strokes


def smooth_stroke(s: np.ndarray, radius: int) -> np.ndarray:
    """Moving average along the stroke: removes the pixel staircase, keeps the end points."""
    if radius <= 0 or len(s) < 3:
        return s
    k = np.ones(2 * radius + 1) / (2 * radius + 1)
    pad = np.pad(s, ((radius, radius), (0, 0)), mode="edge")
    out = np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(2)], 1)
    out[0], out[-1] = s[0], s[-1]
    return out.astype(np.float32)


def thick_fill_strokes(ink: np.ndarray, pen_px: float) -> list[np.ndarray]:
    """Nested outlines, one pen width apart, filling the parts of the ink that are wider than
    about two pen widths. Thinner lines are fully covered by their centre line already."""
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    thick = dist > pen_px  # half-width > 1 pen -> line wider than 2 pens
    if not thick.any():
        return []
    r = int(np.ceil(pen_px))
    region = (cv2.dilate(thick.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0) & ink
    strokes = []
    k = 0
    while True:
        level = ((dist > pen_px * (0.5 + k)) & region).astype(np.uint8)
        if not level.any():
            return strokes
        contours, _ = cv2.findContours(level, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            pts = cnt.reshape(-1, 2)[:, ::-1].astype(np.float32)  # (x, y) -> (row, col)
            strokes.append(np.concatenate([pts, pts[:1]]))
        k += 1


def lineart_strokes(gray: np.ndarray) -> list[np.ndarray]:
    """Trace each drawn line once, along its centre (plus fill for very thick ink)."""
    ink = lineart_mask(gray)
    skel = prune_spurs(remove_staircases(thin(ink)), LINEART_SPUR_PX)
    strokes = [smooth_stroke(s, LINEART_SMOOTH_PX) for s in skeleton_strokes(skel)]
    if LINEART_FILL_THICK:
        strokes += thick_fill_strokes(ink, LINEART_PEN_MM / mm_per_pixel(gray.shape))
    return strokes


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


def scanlines(mask: np.ndarray, angle_deg: float, spacing_px: float,
              rng: random.Random, jitter_px: float):
    """Cut the mask with parallel lines. Returns (lines, d): lines[k] is the list of runs on
    scanline k, each run [start_xy, end_xy] pointing along the unit direction d."""
    h, w = mask.shape
    a = np.deg2rad(angle_deg)
    d = np.array([np.cos(a), np.sin(a)])      # along the lines (x, y)
    nrm = np.array([-d[1], d[0]])              # across the lines
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
    t = np.arange((corners @ d).min(), (corners @ d).max(), 0.5)
    lines = []
    for o in np.arange((corners @ nrm).min() + spacing_px / 2, (corners @ nrm).max(), spacing_px):
        if jitter_px:
            o += rng.uniform(-jitter_px, jitter_px)
        x, y = nrm[0] * o + d[0] * t, nrm[1] * o + d[1] * t
        xi, yi = np.round(x).astype(int), np.round(y).astype(int)
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        on = np.zeros(len(t), bool)
        on[inside] = mask[yi[inside], xi[inside]]
        edges = np.diff(np.r_[0, on.astype(np.int8), 0])
        runs = []
        for s, e in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1):
            if e <= s:
                continue
            p0, p1 = np.array([x[s], y[s]]), np.array([x[e], y[e]])
            if jitter_px:  # ragged ends, but never past the region edge
                limit = min(jitter_px, np.linalg.norm(p1 - p0) / 4)
                p0, p1 = p0 + d * rng.uniform(0, limit), p1 - d * rng.uniform(0, limit)
            runs.append([p0, p1])
        lines.append(runs)
    return lines, d


def chain_runs(lines, d: np.ndarray) -> list[list]:
    """Group runs of consecutive scanlines that overlap along the fill direction into chains:
    each chain is a strip of the region that can be filled in one go."""
    used = [[False] * len(runs) for runs in lines]
    chains = []
    for k, runs in enumerate(lines):
        for i in range(len(runs)):
            if used[k][i]:
                continue
            used[k][i] = True
            chain, cur, kk = [runs[i]], runs[i], k
            while kk + 1 < len(lines):
                a0, a1 = cur[0] @ d, cur[1] @ d
                best = None
                for j, (q0, q1) in enumerate(lines[kk + 1]):
                    if used[kk + 1][j]:
                        continue
                    overlap = min(a1, q1 @ d) - max(a0, q0 @ d)
                    if overlap > 0 and (best is None or overlap > best[0]):
                        best = (overlap, j)
                if best is None:
                    break
                used[kk + 1][best[1]] = True
                cur = lines[kk + 1][best[1]]
                chain.append(cur)
                kk += 1
            chains.append(chain)
    return chains


def chained_strokes(lines, d: np.ndarray, mask: np.ndarray, spacing_px: float,
                    style: str) -> list[np.ndarray]:
    """Turn chains of scanline runs into continuous pen paths. Every connecting move is
    checked against the region; where one would leave it (and draw over white areas),
    the path is split instead.

      zigzag      V V V: only diagonals, edge to opposite edge, one line further each time
      z-lines     Z Z Z: each line drawn in the same direction, diagonal back to the next start
      serpentine  back-and-forth lines joined by short (optionally rounded) turns
    """
    h, w = mask.shape
    ok_mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # rounding slack

    def inside(pts):
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        return bool(ok_mask[yi, xi].all())

    def segment_ok(p, q):
        n = max(2, int(np.linalg.norm(q - p) / 0.5))
        return inside(p + (q - p) * np.linspace(0, 1, n)[:, None])

    def turn(p, q, along, bulge):
        """Serpentine turn: points strictly between p and q, or None if impossible."""
        if FILL_ROUND_TURNS and bulge > 0.3:
            c = (p + q) / 2 + along * 2 * bulge      # quadratic bezier peak reaches `bulge`
            s = np.linspace(0, 1, 24)[:, None]
            curve = (1 - s) ** 2 * p + 2 * (1 - s) * s * c + s ** 2 * q
            if inside(curve):
                return curve[3:-3:4]
        return np.empty((0, 2)) if segment_ok(p, q) else None

    strokes = []
    for chain in chain_runs(lines, d):
        if style == "zigzag":
            if len(chain) == 1:           # a lone line: just draw it
                strokes.append(np.array(chain[0]))
                continue
            # vertices alternate between the two edges: start of even lines, end of odd lines
            verts = [run[0] if i % 2 == 0 else run[1] for i, run in enumerate(chain)]
            path = [verts[0]]
            for v in verts[1:]:
                if not segment_ok(path[-1], v):
                    if len(path) > 1:
                        strokes.append(np.array(path))
                    path = []
                path.append(v)
            if len(path) > 1:
                strokes.append(np.array(path))

        elif style == "z-lines":
            path = [chain[0][0], chain[0][1]]
            for run in chain[1:]:
                if not segment_ok(path[-1], run[0]):   # the diagonal back across
                    strokes.append(np.array(path))
                    path = []
                path += [run[0], run[1]]
            strokes.append(np.array(path))

        else:  # serpentine
            # with rounded turns, pull run ends inwards so the loop can bulge back to the edge
            inset_px = spacing_px / 2 if FILL_ROUND_TURNS else 0.0
            runs, insets = [], []
            for p0, p1 in chain:
                k = min(inset_px, np.linalg.norm(p1 - p0) / 3)
                runs.append((p0 + d * k, p1 - d * k))
                insets.append(k)
            path = [runs[0][0], runs[0][1]]
            forward = True
            for i in range(1, len(runs)):
                q0, q1 = runs[i]
                start, finish = (q1, q0) if forward else (q0, q1)
                conn = turn(path[-1], start, d if forward else -d, min(insets[i - 1], insets[i]))
                if conn is None:
                    strokes.append(np.array(path))
                    path = [start, finish]
                else:
                    path += list(conn) + [start, finish]
                forward = not forward
            strokes.append(np.array(path))
    return strokes


def region_axis(mask: np.ndarray) -> float | None:
    """Long-axis angle (degrees, image coords) of a region from its second moments, or None
    if the region is too round for a meaningful direction (then the layer angle is kept)."""
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    mu20, mu02, mu11 = m["mu20"], m["mu02"], m["mu11"]
    spread = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    major, minor = (mu20 + mu02 + spread) / 2, (mu20 + mu02 - spread) / 2
    if minor <= 0 or np.sqrt(major / minor) < FILL_SHAPE_MIN_ELONGATION:
        return None
    return float(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)))


def fill_strokes(tones: np.ndarray, px_per_mm: float,
                 labels: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Shading: fill each tone region (per pen colour) with hatch lines or a zig-zag.
    Returns (pen index, (row, col) polyline) pairs."""
    rng = random.Random(FILL_SEED)  # stdlib: numpy.random's DLL is blocked by Application Control here
    kernel = np.ones((3, 3), np.uint8)
    jitter_px = max(0.0, FILL_JITTER_MM) * px_per_mm
    base_angle = HATCH_LAYERS[0][0] if HATCH_LAYERS else 0
    strokes = []
    for angle_deg, spacing_mm, max_tone in HATCH_LAYERS:
        mask = cv2.morphologyEx((tones <= max_tone).astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
        spacing_px = max(spacing_mm, 0.3) * px_per_mm
        for pen in np.unique(labels[mask]):
            region = (mask & (labels == pen)).astype(np.uint8)
            n, comp, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < HATCH_MIN_REGION_PX:
                    continue
                x0, y0, w, h = stats[i, :4]
                sub = comp[y0:y0 + h, x0:x0 + w] == i
                angle = angle_deg
                if FILL_DIRECTION == "shape":
                    axis = region_axis(sub)
                    if axis is not None:
                        angle = axis + (angle_deg - base_angle)
                lines, d = scanlines(sub, angle, spacing_px, rng, jitter_px)
                if FILL_STYLE == "lines":
                    polys = [np.array(run) for runs in lines for run in runs]
                else:
                    polys = chained_strokes(lines, d, sub, spacing_px, FILL_STYLE)
                for p in polys:  # crop (x, y) -> image (row, col)
                    strokes.append((int(pen), np.column_stack([p[:, 1] + y0, p[:, 0] + x0])
                                    .astype(np.float32)))
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
        strokes += fill_strokes(tones, 1.0 / mm_per_pixel(gray.shape), labels)
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


def stroke_to_mm(s: np.ndarray, shape, tolerance_mm: float | None = None,
                 min_length_mm: float | None = None) -> np.ndarray | None:
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
    tol = SIMPLIFY_TOLERANCE_MM if tolerance_mm is None else tolerance_mm
    min_len = MIN_LINE_LENGTH_MM if min_length_mm is None else min_length_mm
    pts = cv2.approxPolyDP(pts.reshape(-1, 1, 2), tol, False).reshape(-1, 2)
    if len(pts) < 2 or np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() < min_len:
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
    gray, bgr = prepare(img, LINEART_RESOLUTION_PX if style == "lineart" else None)
    cols = active_colours()
    labels = colour_labels(bgr)

    comic = None
    if style == "comic":
        pixel_strokes, comic = comic_strokes(gray, labels)
    elif style == "lineart":
        pixel_strokes = [p for s in lineart_strokes(gray) for p in split_by_colour(s, labels)]
    else:
        pixel_strokes = [p for s in edge_strokes(gray) for p in split_by_colour(s, labels)]

    layers = {c: [] for c in cols}
    for label, s in pixel_strokes:
        if style == "lineart":
            pts = stroke_to_mm(s, gray.shape, LINEART_SIMPLIFY_MM, LINEART_MIN_LENGTH_MM)
        else:
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
    # line art is judged against the source drawing, so show the real pen width
    # (other styles stay 1 px: thick lines would hide the hatch spacing)
    width = max(1, round(LINEART_PEN_MM * px_per_mm)) if result.style == "lineart" else 1
    for colour, strokes in result.layers.items():
        bgr = tuple(int(v) for v in PEN_COLOURS.get(colour, (0, 0, 0)))
        for s in strokes:
            pts = np.column_stack([s[:, 0] * px_per_mm + margin, h - margin - s[:, 1] * px_per_mm])
            cv2.polylines(canvas, [pts.round().astype(np.int32)], False, bgr, width, cv2.LINE_AA)
    return canvas


def main():
    global IMAGE_WIDTH_MM, ACTIVE_COLOURS
    parser = argparse.ArgumentParser(description="Convert an image to a URScript line drawing.")
    parser.add_argument("image", type=Path, help="input image file, e.g. monalisa.png")
    parser.add_argument("-o", "--output", type=Path, help="output .script (default: <image>.script)")
    parser.add_argument("--width", type=float, help=f"drawing width in mm (default {IMAGE_WIDTH_MM})")
    parser.add_argument("--style", choices=["edges", "comic", "lineart"], default="edges",
                        help="edges = outlines only, comic = ink lines + shading, "
                             "lineart = trace an existing line drawing (default edges)")
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
