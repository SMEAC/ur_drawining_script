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


def send_script(host: str, port: int, text: str, timeout: float = 5.0) -> None:
    """Send a multi-line URScript program, as in the UR guide's example: connect, send the
    program plus a final newline, close. Raises OSError if the robot can't be reached."""
    with socket.create_connection((host, int(port)), timeout=timeout) as s:
        s.sendall((text + "\n").encode("utf-8"))
