"""
Online robot control: send a drawing straight to a UR controller over TCP/IP,
instead of saving .script files and loading them in PolyScope.

Follows the "Sending Multiple Line Scripts" example of the Universal Robots guide
"UR remote control via TCP/IP" (forum.universal-robots.com/t/32973): a complete
`def ...(): ... end` program is sent as text to the primary (30001) or secondary
(30002) interface, and the controller runs it immediately. A program sent later
replaces the running one, which is how Stop works.

A program sent this way runs outside PolyScope's generated program, so installation
features such as Feature1 are not defined. The canvas pose (Feature1 in the robot
base frame) is therefore written into the program as a literal; the .script files
log it as "Feature Pose: p[...]" when they run.

Requirements on the robot: e-Series robots only accept external scripts in Remote
Control mode (Settings > System > Remote Control, then switch the pendant to Remote).
"""

import re
import socket

import image_to_urscript as core

DEFAULT_PORT = 30001
PROGRAM_NAME = "robot_painter"

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_POSE_RE = re.compile(r"^\s*p?\s*\[\s*(" + _NUM + r")" + (r"\s*,\s*(" + _NUM + r")") * 5 + r"\s*\]\s*$")


def parse_pose(text: str) -> list[float]:
    """'p[x, y, z, rx, ry, rz]' (metres / radians, as UR prints poses) -> 6 floats."""
    m = _POSE_RE.match(text or "")
    if not m:
        raise ValueError("Canvas pose must look like p[x, y, z, rx, ry, rz] (metres, radians)")
    pose = [float(v) for v in m.groups()]
    if all(v == 0 for v in pose):
        raise ValueError("Canvas pose is all zeros: enter the Feature1 pose in the robot base frame")
    return pose


def pose_text(pose) -> str:
    return "p[" + ", ".join(f"{v:.6f}" for v in pose) + "]"


def program_text(result, colour: str, feature_pose, speed_factor: float = 1.0) -> str:
    """One self-contained URScript program (def ... end) that draws the strokes of one pen,
    relative to the canvas pose. speed_factor scales the draw and travel speeds."""
    strokes = result.layers.get(colour, [])
    m = core.m
    draw_v = core.DRAW_SPEED * speed_factor
    travel_v = core.TRAVEL_SPEED * speed_factor
    n_points = sum(len(s) for s in strokes)
    lines = [
        f"def {PROGRAM_NAME}():",
        f"  # {colour} pen: {len(strokes)} strokes, {n_points} points, "
        f"{result.width_mm:.1f} x {result.height_mm:.1f} mm, speed {speed_factor:.0%}",
        f"  global feature = {pose_text(feature_pose)}",
        f"  global draw_v = {draw_v:.4f}",
        f"  global travel_v = {travel_v:.4f}",
        f"  global acc = {core.ACCELERATION}",
        f"  global z_down = {m(core.PEN_DOWN_Z_MM)}",
        f"  global z_up = {m(core.PEN_UP_Z_MM)}",
        f"  global rx = {core.ORIENT_RX}",
        f"  global ry = {core.ORIENT_RY}",
        f"  global rz = {core.ORIENT_RZ}",
        # short helper names keep a drawing with thousands of points compact to send
        "  def P(x, y):",  # lift, travel to (x, y), lower the pen
        "    movel(pose_trans(feature, p[x, y, z_up, rx, ry, rz]), a=acc, v=travel_v)",
        "    movel(pose_trans(feature, p[x, y, z_down, rx, ry, rz]), a=acc, v=draw_v)",
        "  end",
        "  def L(x, y, r):",  # draw to (x, y) with blend radius r
        "    movel(pose_trans(feature, p[x, y, z_down, rx, ry, rz]), a=acc, v=draw_v, r=r)",
        "  end",
        "  def U():",  # lift the pen where it is
        "    here = pose_trans(pose_inv(feature), get_actual_tcp_pose())",
        "    movel(pose_trans(feature, p[here[0], here[1], z_up, rx, ry, rz]), a=acc, v=travel_v)",
        "  end",
        f'  textmsg("Robot Painter: drawing {colour}, {len(strokes)} strokes")',
    ]
    for s in strokes:
        radii = core.blend_radii(s)
        lines.append(f"  P({m(s[0][0])}, {m(s[0][1])})")
        for (x, y), r in zip(s[1:], radii[1:]):
            lines.append(f"  L({m(x)}, {m(y)}, {m(r)})")
        lines.append("  U()")
    lines += [f'  textmsg("Robot Painter: {colour} finished")', "end", ""]
    return "\n".join(lines)


def stop_program_text(feature_pose=None) -> str:
    """Replaces the running drawing: decelerate, then lift the pen off the canvas
    (the lift needs the canvas pose; without it the robot only stops)."""
    if feature_pose is None:
        return "def robot_painter_stop():\n  stopl(1.0)\n  textmsg(\"Robot Painter: stopped\")\nend\n"
    return "\n".join([
        "def robot_painter_stop():",
        f"  global feature = {pose_text(feature_pose)}",
        "  stopl(1.0)",
        "  here = pose_trans(pose_inv(feature), get_actual_tcp_pose())",
        f"  movel(pose_trans(feature, p[here[0], here[1], {core.m(core.PEN_UP_Z_MM)}, "
        f"{core.ORIENT_RX}, {core.ORIENT_RY}, {core.ORIENT_RZ}]), a=0.5, v=0.05)",
        '  textmsg("Robot Painter: stopped")',
        "end",
        "",
    ])


JOG_SPEED = 0.02   # m/s, slow on purpose: jogging is for testing
JOG_ACC = 0.2      # m/s^2


def jog_program_text(dx_mm: float, dy_mm: float, dz_mm: float, feature_pose=None) -> str:
    """Move the tool in a straight line by (dx, dy, dz) mm, keeping its orientation.
    With a canvas pose the step is along the canvas (Feature1) axes, the same axes the
    drawing uses; without one it is along the robot base axes."""
    d = f"p[{dx_mm / 1000:.5f}, {dy_mm / 1000:.5f}, {dz_mm / 1000:.5f}, 0, 0, 0]"
    lines = ["def robot_painter_jog():", "  here = get_actual_tcp_pose()"]
    if feature_pose is None:
        lines.append(f"  target = pose_add(here, {d})")           # base-frame translation
    else:
        lines += [f"  feature = {pose_text(feature_pose)}",
                  # translate the TCP in the canvas frame: feature * D * (feature^-1 * here)
                  f"  target = pose_trans(feature, pose_trans({d}, pose_trans(pose_inv(feature), here)))"]
    lines += [f"  movel(target, a={JOG_ACC}, v={JOG_SPEED})",
              f'  textmsg("Robot Painter: jog {dx_mm:+g} {dy_mm:+g} {dz_mm:+g} mm '
              f'({"base" if feature_pose is None else "canvas"} frame)")',
              "end", ""]
    return "\n".join(lines)


GOTO_SPEED = 0.05  # m/s for the move to the canvas origin (can be a long move)


def goto_origin_program_text(feature_pose, height_mm: float = 10.0) -> str:
    """Move in a straight line to `height_mm` above the canvas origin (Feature1 0,0,0), in
    the pen-up direction and with the drawing's tool orientation: a quick visual check of
    the canvas pose, the TCP and the orientation before drawing."""
    up = 1.0 if core.PEN_UP_Z_MM >= core.PEN_DOWN_Z_MM else -1.0   # pen-up side of the canvas
    z = core.m(core.PEN_DOWN_Z_MM + up * height_mm)
    return "\n".join([
        "def robot_painter_goto_origin():",
        f"  feature = {pose_text(feature_pose)}",
        f"  target = pose_trans(feature, p[0, 0, {z}, {core.ORIENT_RX}, {core.ORIENT_RY}, {core.ORIENT_RZ}])",
        f'  textmsg("Robot Painter: moving to {height_mm:g} mm above the canvas origin")',
        f"  movel(target, a={JOG_ACC}, v={GOTO_SPEED})",
        f'  textmsg("Robot Painter: at {height_mm:g} mm above the canvas origin: ", target)',
        "end", "",
    ])


def pose_log_program_text(feature_pose=None) -> str:
    """Write the current TCP pose to the pendant's Log tab and show it in a popup; no motion.
    Feature1 itself can't be logged: programs sent over the socket run outside PolyScope's
    program, where installation features are undefined (see urscript/print_feature_pose.script)."""
    lines = ["def robot_painter_pose():",
             "  tcp = get_actual_tcp_pose()",
             '  textmsg("Robot Painter TCP pose (base frame): ", tcp)']
    msg = 'str_cat("TCP pose, base frame (m, rad): ", tcp)'
    if feature_pose is not None:
        lines += [f"  feature = {pose_text(feature_pose)}",
                  "  on_canvas = pose_trans(pose_inv(feature), tcp)",
                  '  textmsg("Robot Painter TCP pose (canvas frame): ", on_canvas)']
        msg = f'str_cat({msg}, str_cat("<br>Canvas frame: ", on_canvas))'
    lines += [f'  popup({msg}, title="Robot Painter", blocking=False)', "end", ""]
    return "\n".join(lines)


def pose_matrix(pose):
    """UR pose [x, y, z, rx, ry, rz] (rotation vector) -> 4x4 homogeneous matrix."""
    import cv2
    import numpy as np
    m = np.eye(4)
    m[:3, :3] = cv2.Rodrigues(np.array(pose[3:], float))[0]
    m[:3, 3] = pose[:3]
    return m


def orientation_in_canvas(tcp, feature_pose) -> list[float]:
    """Rotation vector of the tool relative to the canvas (Feature1) frame: the values
    ORIENT_RX/RY/RZ must have for the drawing to keep the tool exactly as it is now."""
    import cv2
    import numpy as np
    local = np.linalg.inv(pose_matrix(feature_pose)) @ pose_matrix(tcp)
    return [float(v) for v in cv2.Rodrigues(local[:3, :3])[0].ravel()]


def describe_tcp(tcp, feature_pose=None, origin_height_mm: float = 10.0) -> str:
    """Human-readable TCP position in the base frame and, with a canvas pose, in the canvas
    frame - next to what "Go to canvas origin" aims for - plus how far the tool orientation
    is from the drawing orientation."""
    import cv2
    import numpy as np
    mm = [v * 1000 for v in tcp[:3]]
    lines = [f"Base   x{mm[0]:7.1f} y{mm[1]:7.1f} z{mm[2]:7.1f} mm"]
    if feature_pose is not None:
        local = np.linalg.inv(pose_matrix(feature_pose)) @ pose_matrix(tcp)   # = pose_trans(pose_inv(F), tcp)
        x, y, z = local[:3, 3] * 1000
        up = 1.0 if core.PEN_UP_Z_MM >= core.PEN_DOWN_Z_MM else -1.0
        tz = core.PEN_DOWN_Z_MM + up * origin_height_mm
        wanted = cv2.Rodrigues(np.array([core.ORIENT_RX, core.ORIENT_RY, core.ORIENT_RZ], float))[0]
        cos = (np.trace(wanted.T @ local[:3, :3]) - 1) / 2
        angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        lines += [f"Canvas x{x:7.1f} y{y:7.1f} z{z:7.1f} mm",
                  f"Origin check target x 0 y 0 z {tz:g}:",
                  f"  off by {np.hypot(np.hypot(x, y), z - tz):.1f} mm",
                  f"Tool angle {angle:.1f}° from drawing orientation"]
    return "\n".join(lines)


def read_tcp_pose(host: str, port: int, timeout: float = 3.0) -> list[float]:
    """Read the actual TCP pose from the robot state stream of the primary/secondary
    interface (message type 16 = robot state, sub-package type 4 = Cartesian info:
    X, Y, Z, Rx, Ry, Rz as big-endian doubles). Raises OSError / TimeoutError."""
    import struct
    import time
    deadline = time.monotonic() + timeout
    buf = b""
    with socket.create_connection((host, int(port)), timeout=timeout) as s:
        while time.monotonic() < deadline:
            s.settimeout(max(0.1, deadline - time.monotonic()))
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
            while len(buf) >= 5:
                size, kind = struct.unpack_from(">iB", buf)
                if size < 5 or size > 1_000_000:
                    raise OSError("unexpected data - is this a UR primary/secondary port?")
                if len(buf) < size:
                    break
                msg, buf = buf[:size], buf[size:]
                if kind != 16:          # 20 = version message etc.
                    continue
                pos = 5
                while pos + 5 <= len(msg):
                    sub_size, sub_kind = struct.unpack_from(">iB", msg, pos)
                    if sub_size < 5:
                        break
                    if sub_kind == 4 and sub_size >= 5 + 48:
                        return list(struct.unpack_from(">6d", msg, pos + 5))
                    pos += sub_size
    raise TimeoutError("no Cartesian info in the robot state stream")


def check_connection(host: str, port: int, timeout: float = 3.0) -> int:
    """Connect without sending anything. The primary/secondary interfaces (30001/30002)
    stream robot state as soon as a client connects, so receiving data shows that a UR
    controller is answering. Returns the number of bytes received (0 = connected, silent).
    Raises OSError if the connection fails."""
    with socket.create_connection((host, int(port)), timeout=timeout) as s:
        s.settimeout(timeout)
        try:
            return len(s.recv(4096))
        except socket.timeout:
            return 0


def send_script(host: str, port: int, text: str, timeout: float = 5.0) -> None:
    """Send a multi-line URScript program, as in the UR guide's example: connect, send the
    program plus a final newline, close. Raises OSError if the robot can't be reached."""
    with socket.create_connection((host, int(port)), timeout=timeout) as s:
        s.sendall((text + "\n").encode("utf-8"))
