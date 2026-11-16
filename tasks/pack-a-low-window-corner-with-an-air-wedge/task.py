"""A window being fitted. A white uPVC fixed-light window, 915 x 600 mm, 14 kg, stands in a timber opening built on the
bench (the bench top is the opening's sill). Its right bottom corner already sits on a 6 mm plastic packer; its left
bottom corner is down on the sill, so the window is 6 mm out of level. On the sill in front of it lie an air wedge (a
flat inflatable lifting bag with a squeeze bulb on a hose and a release button) and one loose 6 mm packer.

The job is the fitter's: get the low corner up onto the loose packer. The window is too heavy and has no hold for a
gripper, so the wedge does the lifting: slide the flat bag under the bottom rail, squeeze the bulb until the corner is
clear of the sill, slide the packer under the corner, press the release button to let the window down onto it, and take
the wedge out.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "The window's left corner is sitting low on the sill. Get it level on the loose packer, using the air wedge to lift it, and leave nothing but the packers under the window." \\
           --scorer task_score -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 --max-steps 2400
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from wedgekit import Slider, fixture, start_at  # noqa: E402  (importing wedgekit hooks yam_arena so the fixtures are simulated)
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

WHITE, GLASS, PINE, STEEL, BLUE, BAG, BAG2, ORANGE, RUBBER, RED = (
    (0.93, 0.93, 0.91), (0.70, 0.85, 0.92, 0.25), (0.78, 0.62, 0.40), (0.60, 0.62, 0.65), (0.10, 0.35, 0.85),
    (0.16, 0.17, 0.19), (0.24, 0.25, 0.28), (0.95, 0.45, 0.05), (0.07, 0.07, 0.08), (0.85, 0.07, 0.05))
VB = {"collide": False}

# -- the window in its opening (the bench top is the sill) -------------------------------------------------------------
XF = 0.40  # the frame's room-side face (towards the arms)
DEPTH = 0.07  # the frame's depth, front to back
XC = XF + DEPTH / 2
W, H, PROF = 0.915, 0.60, 0.06  # frame width and height, and the width of its profile's face
YR, YL = -W / 2, W / 2  # the frame's right and left ends
MASS = 14.0
PK = (0.10, 0.028, 0.006)  # a flat packer: 100 x 28 x 6 mm
PIV_Y = YR + 0.03  # the packer under the right corner: the frame rocks on it
LEN = YL - PIV_Y  # from that packer to the frame's left end
LOW = -PK[2] / LEN  # the frame's tilt with its left corner down on the sill (rad)
HEAD_GAP = 0.010  # between the frame's top and the opening's head when the frame is level
POST = (0.089, 0.038)  # 2x4 timber, 38 x 89 mm
SIDE_GAP = 0.010

_oy = 0.5 * (YR + YL) - PIV_Y  # the frame's middle, from the pivot
FRAME = Slider("frame", (0.0, 0.0, PK[2]), "X", (
    ((0, _oy, PROF / 2), (DEPTH, W, PROF), WHITE),  # bottom rail
    ((0, _oy, H - PROF / 2), (DEPTH, W, PROF), WHITE),  # top rail
    ((0, YR - PIV_Y + PROF / 2, H / 2), (DEPTH, PROF, H - 2 * PROF), WHITE),  # right jamb
    ((0, YL - PIV_Y - PROF / 2, H / 2), (DEPTH, PROF, H - 2 * PROF), WHITE),  # left jamb
    ((0, _oy, H / 2), (0.024, W - 2 * PROF, H - 2 * PROF), GLASS)),  # the sealed glass unit
    low=LOW, high=HEAD_GAP / LEN, kind="revolute", damping=40.0, mass=MASS)
_post_h = PK[2] + H + HEAD_GAP
OPENING = fixture("opening", (XC, PIV_Y, TOP), [
    ((-0.03, 0, PK[2] / 2), PK, BLUE),  # the packer already under the right corner (4.5 cm of it showing in front)
    ((0, YR - SIDE_GAP - POST[1] / 2 - PIV_Y, _post_h / 2), (POST[0], POST[1], _post_h), PINE),  # right post
    ((0, YL + SIDE_GAP + POST[1] / 2 - PIV_Y, _post_h / 2), (POST[0], POST[1], _post_h), PINE),  # left post
    ((0, _oy, _post_h + POST[1] / 2), (POST[0], W + 2 * (SIDE_GAP + POST[1]), POST[1]), PINE),  # head
    *[((sx * (POST[0] / 2 + 0.0015), _oy + sy * 0.25, _post_h - 0.005), (0.003, 0.03, 0.07), STEEL, VB)  # retaining plates
      for sx in (-1, 1) for sy in (-1, 1)]], [FRAME], friction=0.5)
start_at("opening", "frame", LOW)

# -- the loose packer, on the sill in front of the low corner -------------------------------------------------------------
PX0, PY0 = 0.31, 0.42
PACKER = fixture("packer", (PX0, PY0, TOP + 0.0005), [((0, 0, PK[2] / 2), PK, BLUE)], [], fixed=False,
                 base_mass=0.017, friction=0.4)

# -- the air wedge: bag 160 x 150 x 2 mm, grip block on its hose socket, 75 mm hose, 80 x 45 mm squeeze bulb, release button --
BAG_L, BAG_W, BAG_T = 0.16, 0.15, 0.002
WX0, WY0 = 0.305, -0.08  # the bag's centre at the start: flat on the sill, its front edge 1.5 cm from the frame
BLOCK = (0.025, 0.020, 0.022)  # the grip block on the hose socket, at the middle of the bag's near edge
HOSE, HOSE_Z = 0.075, 0.006  # the hose runs straight back from the block, towards the arms
BULB_D, BULB_L, NECK = 0.045, 0.08, 0.01
BULB_Z = BULB_D / 2 + 0.0008 - HOSE_Z  # the bulb's centre above the hose's
SQUEEZE = 0.012  # how far the bulb's side goes in on a full squeeze
STEP = 0.0012  # the bag grows this much per squeeze (under this load)
LIFT_MAX = 0.012
VALVE_ON = 0.0015  # the button lets air out once it is this far in (of its 3 mm)
VENT = 0.008  # the bag fills after a squeeze, and sinks while the button is held, at this rate (m/s)
VALVE_X = -0.03  # the release button, on top of the bulb, 2 cm from its hose end
LIFT = Slider("lift", (0, 0, 0), "Z", (((0, 0, BAG_T / 2), (BAG_L - 0.002, BAG_W - 0.002, BAG_T), BAG2),),
              low=0.0, high=0.02, stiffness=4.0e5, damping=7.0e3, mass=0.03, gravity=False,
              pump=("pad", 0.009, 0.003, STEP, "valve", VALVE_ON, VENT, LIFT_MAX))
MID = Slider("mid", (0, 0, 0), "Z", (((0, 0, BAG_T / 2), (BAG_L - 0.012, BAG_W - 0.012, BAG_T), BAG, VB),),
             low=0.0, high=0.02, stiffness=2000.0, damping=20.0, mass=0.004, gravity=False, follows="lift", ratio=0.5)
HK = dict(kind="revolute", low=-1.5, high=1.5, stiffness=2.0, damping=0.05)
_link = (((-HOSE / 4, 0, 0), (HOSE / 2, 0.008, 0.008), RUBBER, {"shape": "cyl", "axis": "X"}),)
# the hose leaves the block through a joint that lets it droop or lift (so the bulb lies on the sill under its own weight
# and a hand holding the bulb does not hold the bag), then two links that let it bend sideways
H0 = Slider("h0", (-BAG_L / 2 - BLOCK[0], 0, HOSE_Z), "Y", (((0, 0, 0), (0.006, 0.006, 0.006), RUBBER, VB),),
            kind="revolute", low=-0.6, high=0.6, stiffness=0.02, damping=0.01, mass=0.002)
H1 = Slider("h1", (0, 0, 0), "Z", _link, parent="h0", mass=0.004, **HK)
H2 = Slider("h2", (-HOSE / 2, 0, 0), "Z", _link, parent="h1", mass=0.004, **HK)
_bx = -NECK - BULB_L / 2  # the bulb's middle, from the end of the hose
BULB = Slider("bulb", (-HOSE / 2, 0, 0), "Z", (
    ((-NECK / 2, 0, 0), (NECK, 0.012, 0.012), RUBBER, {"shape": "cyl", "axis": "X", "collide": False}),  # neck
    ((_bx, 0, BULB_Z), (BULB_L, BULB_D, BULB_D), RUBBER, {"shape": "cyl", "axis": "X", "collide": False}),  # the bulb (as seen)
    ((_bx, -0.0075, BULB_Z), (BULB_L, 0.030, 0.044), RUBBER)),  # its body (as felt)
    parent="h2", mass=0.03, **HK)
PAD = Slider("pad", (_bx, 0.015, BULB_Z), "Y", (((0, 0, 0), (0.06, 0.015, 0.02), RUBBER),),
             parent="bulb", low=-SQUEEZE, high=0.0, stiffness=20.0 / SQUEEZE, damping=4.0, mass=0.004, gravity=False)
VALVE = Slider("valve", (VALVE_X, 0, BULB_Z + BULB_D / 2), "Z", (
    ((0, 0, 0.002), (0.014, 0.014, 0.004), RED, {"shape": "cyl", "axis": "Z"}),),
    parent="bulb", low=-0.003, high=0.0, stiffness=3.0 / 0.003, damping=3.0, mass=0.002, gravity=False)
WEDGE = fixture("wedge", (WX0, WY0, TOP + 0.0005), [
    ((0, 0, BAG_T / 2), (BAG_L, BAG_W, BAG_T), BAG),
    ((-BAG_L / 2 - BLOCK[0] / 2, 0, BLOCK[2] / 2), BLOCK, ORANGE)],
    [LIFT, MID, H0, H1, H2, BULB, PAD, VALVE], fixed=False, base_mass=0.12, friction=0.5)
OBJECTS = [OPENING, PACKER, WEDGE]

# -- scoring ---------------------------------------------------------------------------------------------------------------
CLEAR = 0.05  # "let go": both grippers this far from the frame, the packer and the wedge
ON_PACKER = 0.0015  # the corner is on the packer when it is within 1.5 mm of the packer's 6 mm
LIFTED = 0.003
UNDER_MIN = 0.04  # at least 4 cm of the packer's length under the frame
ZONE = PROF  # ... and its centre line within 6 cm of the frame's left end
_TRACE = os.environ.get("TASK_TRACE")
_EVERY = int(os.environ.get("TRACE_EVERY", 10))
_steps = 0


def _yaw(q) -> float:
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def corner(objects) -> float:
    """The frame's left bottom corner above the sill (m): 0 down on the sill, 6 mm level."""
    return float(PK[2] + LEN * objects["opening"].q["frame"])


def packer_under(objects) -> bool:
    """The loose packer lies flat under the frame's left corner: 4 cm or more of its length under the frame, its centre
    line 1.4 to 6 cm in from the frame's left end, lengthwise across the frame (within 20 degrees)."""
    p = objects["packer"]
    x, y, z = (float(v) for v in p.pos)
    a = _yaw(p.quat_xyzw)
    a = (a + math.pi / 2) % math.pi - math.pi / 2
    half = PK[0] / 2 * math.cos(a)
    under = min(x + half, XF + DEPTH) - max(x - half, XF)
    return bool(under >= UNDER_MIN and YL - ZONE <= y <= YL - PK[1] / 2 and abs(a) <= math.radians(20)
                and abs(z - TOP) <= 0.004)


def _bag_corners(objects, block=False) -> np.ndarray:
    """The bag's four corners on the sill (with ``block``: the rectangle stretched to take in the grip block)."""
    w = objects["wedge"]
    a = _yaw(w.quat_xyzw)
    c, s = math.cos(a), math.sin(a)
    near = -BAG_L / 2 - (BLOCK[0] if block else 0.0)
    pts = np.array([(x, sy * BAG_W / 2) for x in (near, BAG_L / 2) for sy in (-1, 1)])
    return np.asarray(w.pos, float)[:2] + pts @ np.array([[c, s], [-s, c]])


def wedge_out(objects) -> bool:
    """No part of the bag is under the frame: the whole bag is on the room side of the frame's face."""
    return bool(_bag_corners(objects)[:, 0].max() <= XF)


def _to_segment(p, a, b) -> float:
    ab = b - a
    u = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + u * ab)))


def hands_clear(objects) -> bool:
    """Both grippers' tool points are 5 cm or more from the frame's face, the loose packer, the bag and its grip block,
    the hose and the bulb."""
    w, p = objects["wedge"], objects["packer"]
    b = {k: np.asarray(v, float) for k, v in w.bodies.items()}
    line = [b[k] for k in ("h1", "h2", "bulb", "pad") if k in b]  # along the hose to the bulb's middle
    if len(line) >= 2:
        d = line[-1] - line[-2]
        line.append(line[-1] + d / max(float(np.linalg.norm(d)), 1e-9) * BULB_L / 2)  # ... and on to its far end
    a = _yaw(p.quat_xyzw)
    half = np.array([math.cos(a), math.sin(a), 0.0]) * PK[0] / 2
    pk = np.asarray(p.pos, float)
    rect = _bag_corners(objects, block=True)
    yaw = _yaw(w.quat_xyzw)
    centre = rect.mean(axis=0)
    ext = np.array([(BAG_L + BLOCK[0]) / 2, BAG_W / 2])
    for h in w.hands.values():
        t = np.asarray(h["tcp"], float)
        if t[0] > XF - CLEAR:  # the frame
            return False
        if _to_segment(t, pk - half, pk + half) < CLEAR + PK[1] / 2:  # the loose packer
            return False
        if any(_to_segment(t, q0, q1) < CLEAR + BULB_D / 2 for q0, q1 in zip(line, line[1:])):  # hose and bulb
            return False
        local = np.array([math.cos(yaw) * (t[0] - centre[0]) + math.sin(yaw) * (t[1] - centre[1]),
                          -math.sin(yaw) * (t[0] - centre[0]) + math.cos(yaw) * (t[1] - centre[1])])
        gap = np.r_[np.maximum(np.abs(local) - ext, 0.0), max(t[2] - TOP - BLOCK[2], 0.0)]
        if np.linalg.norm(gap) < CLEAR:  # the bag and its grip block
            return False
    return True


def score(objects):
    """1.0: the corner rests on the loose packer (6 mm up, within 1.5 mm), the wedge is out from under the frame and both
    grippers have let go. 0.75: on the packer, but the wedge is still under the frame or a gripper has not let go.
    0.5: the packer is under the corner but the corner is still held more than 1.5 mm above it. 0.25: the corner is
    3 mm or more off the sill, with no packer under it. 0 otherwise."""
    global _steps
    _steps += 1
    c, under = corner(objects), packer_under(objects)
    if under and abs(c - PK[2]) <= ON_PACKER:
        s = 1.0 if wedge_out(objects) and hands_clear(objects) else 0.75
    elif under and c > PK[2] + ON_PACKER:
        s = 0.5
    elif c >= LIFTED:
        s = 0.25
    else:
        s = 0.0
    if _TRACE and _steps % _EVERY == 0:
        w, p = objects["wedge"], objects["packer"]
        with open(_TRACE, "a") as f:
            f.write(json.dumps({
                "call": _steps, "corner_mm": round(c * 1000, 2), "lift_mm": round(w.q["lift"] * 1000, 2),
                "frame_speed": round(objects["opening"].speed["frame"], 4),
                "pump": {k: round(v, 4) if isinstance(v, float) else v for k, v in w.pumped.get("lift", {}).items()},
                "pad_mm": round(w.q["pad"] * 1000, 1), "valve_mm": round(w.q["valve"] * 1000, 1),
                "packer": [round(float(v), 4) for v in p.pos], "packer_under": under,
                "bag": [round(float(v), 4) for v in w.pos], "bag_yaw_deg": round(math.degrees(_yaw(w.quat_xyzw)), 1),
                "bulb": [round(float(v), 3) for v in w.bodies.get("bulb", (0, 0, 0))],
                "wedge_out": wedge_out(objects), "clear": hands_clear(objects),
                "hands": {a: [round(float(v), 3) for v in h["tcp"]] + [round(h["open"], 2)] for a, h in w.hands.items()},
                "score": s}) + "\n")
    return s


def success(objects):
    return bool(score(objects) == 1.0)


def distance(objects):
    return 0.0 if score(objects) == 1.0 else abs(corner(objects) - PK[2]) + 0.01


# -- scripted demo (open loop: fixed waypoints, nothing read back from the simulator) ----------------------------------------
YAW = math.pi / 2
PUSH = 0.085  # the bag goes in until its far edge is level with the frame's back
SHIFT = 0.007  # squeezing the bulb from both sides leaves it this far towards its soft side
PULL = 0.11
SQUEEZES = int(os.environ.get("SQUEEZES", 5))


def demo():
    """Right arm: grip the wedge's block and slide the bag under the bottom rail, squeeze the bulb five times, then, once
    the left arm has pushed the packer under the corner, press the release button and pull the wedge out by its block.
    Left arm: push the loose packer under the raised corner with one fingertip of its open hand riding on the sill."""
    c = FINGER_CLOSED
    g_block = 0.5 * FINGER_OPEN  # 47 mm: round the 20 mm block
    g_bulb = 0.8 * FINGER_OPEN  # 75 mm: round the 45 mm bulb
    up = TOP + 0.10
    bx, by, bz = WX0 - BAG_L / 2 - BLOCK[0] / 2, WY0, TOP + 0.017  # the block; fingertips 4 mm above the sill
    ux, uy = WX0 - BAG_L / 2 - BLOCK[0] - HOSE - NECK - BULB_L / 2 + PUSH, WY0  # the bulb's middle, after the push
    uz = TOP + BULB_D / 2
    vx = WX0 - BAG_L / 2 - BLOCK[0] - HOSE + VALVE_X + PUSH  # the release button, after the push
    vz = TOP + BULB_D + 0.004 + 0.013 - 0.012  # fingertips aimed 12 mm below the button's top: it goes in its 3 mm

    r = [Waypoint(2.5, (bx, by, up), g_block, "above the wedge's grip block"),
         Waypoint(1.5, (bx, by, bz), g_block, "down round it"),
         Waypoint(0.7, (bx, by, bz), c, "grip it"),
         Waypoint(3.0, (bx + PUSH, by, bz), c, "slide the bag under the bottom rail"),
         Waypoint(0.6, (bx + PUSH, by, bz), g_block, "let go"),
         Waypoint(1.0, (bx + PUSH, by, up), g_block, "up"),
         Waypoint(2.0, (ux, uy, up), g_bulb, "over the bulb"),
         Waypoint(1.5, (ux, uy, uz), g_bulb, "down round the bulb")]
    for k in range(SQUEEZES):
        r += [Waypoint(0.6, (ux, uy, uz), c, f"squeeze {k + 1}"),
              Waypoint(0.6, (ux, uy, uz), g_bulb, "let it fill")]
    t_up = sum(w.duration for w in r)  # the corner is up
    r += [Waypoint(1.0, (ux, uy, up), g_bulb, "up"),
          Waypoint(1.5, (vx, uy + SHIFT, up), c, "over the release button, fingers closed", YAW)]

    # the left hand, open, turned so one finger leads: that fingertip pushes the packer, 5 cm ahead of the wrist
    lead = -FINGER_OPEN + 0.004  # the leading finger's front face, 51 mm ahead of the tool point
    px, py, pz = PX0 - PK[0] / 2 - 0.008 - lead, PY0, TOP + 0.013  # behind the packer, fingertips on the sill
    rest_l = (0.15, 0.50, TOP + 0.16)  # out to the side, clear of the camera's view of the low corner
    left = [Waypoint(2.5, rest_l, FINGER_OPEN, "wait, fingers open", YAW),
            Waypoint(t_up - 2.5, rest_l, FINGER_OPEN, "wait for the corner to come up", YAW),
            Waypoint(1.5, (px, py, TOP + 0.08), FINGER_OPEN, "behind the packer", YAW),
            Waypoint(1.2, (px, py, pz), FINGER_OPEN, "fingertips down on the sill", YAW),
            Waypoint(4.0, (XF - 0.052 - lead, py, pz), FINGER_OPEN, "push the packer under the corner", YAW),
            Waypoint(0.8, (XF - 0.09 - lead, py, TOP + 0.04), FINGER_OPEN, "back off", YAW),
            Waypoint(1.5, rest_l, FINGER_OPEN, "away", YAW)]
    t_packed = sum(w.duration for w in left)

    r += [Waypoint(max(0.1, t_packed - sum(w.duration for w in r)), (vx, uy + SHIFT, up), c, "wait for the packer", YAW),
          Waypoint(1.0, (vx, uy + SHIFT, vz), c, "press the release button", YAW),
          Waypoint(2.5, (vx, uy + SHIFT, vz), c, "hold it while the window comes down", YAW),
          Waypoint(1.0, (vx, uy + SHIFT, up), c, "up", YAW),
          Waypoint(1.5, (bx + PUSH, by, up), g_block, "above the grip block"),
          Waypoint(1.2, (bx + PUSH, by, bz), g_block, "down round it"),
          Waypoint(0.7, (bx + PUSH, by, bz), c, "grip it"),
          Waypoint(3.0, (bx + PUSH - PULL, by, bz), c, "pull the wedge out"),
          Waypoint(0.6, (bx + PUSH - PULL, by, bz), g_block, "let go"),
          Waypoint(0.5, (bx + PUSH - PULL, by, TOP + 0.05), g_block, "up clear of the block"),
          Waypoint(1.2, (0.20, -0.27, TOP + 0.16), g_block, "away to the side, so the wedge is seen lying free")]
    T = max(sum(w.duration for w in left), sum(w.duration for w in r)) + 3.0
    for arm in (left, r):
        arm.append(Waypoint(T - sum(w.duration for w in arm), arm[-1].pos, arm[-1].finger, "wait", arm[-1].yaw))
    return {"left": left, "right": r}
