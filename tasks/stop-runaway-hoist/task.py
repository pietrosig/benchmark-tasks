"""A workshop bench under an overhead electric hoist. The hoist is lowering a large steel toolbox (25 kg) onto the bench at its slow
speed (1.5 cm/s), and a 500 mL glass bottle of acid has been left standing where the toolbox will land. The hoist's
pendant control hangs from its cable in front of the bench; its down button has jammed, so the hoist will not stop by
itself. The pendant's red emergency-stop mushroom stops it (latching, 4 mm of travel). The pendant hangs free: push the
mushroom without steadying the pendant and it swings away instead of being pressed.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "The hoist won't stop lowering the toolbox onto the bottle of acid. Stop it with the emergency stop on its pendant." \\
           --scorer task_score -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 --max-steps 400

ROUGH=1 (demo only): one hand jabs the mushroom with closed fingers, without holding the pendant.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from hoistkit import Slider, fixture  # noqa: E402  (importing hoistkit hooks yam_arena so the fixtures are simulated)
from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 0.5
SCORE_OVER = "end"
ROUGH = int(os.environ.get("ROUGH", 0))

STEEL_RED, STEEL, DARK, YELLOW, RED, BLACK, GREEN, AMBER, CAP_C, MAT, CABLE = (
    (0.70, 0.12, 0.10), (0.55, 0.57, 0.60), (0.15, 0.15, 0.16), (0.96, 0.80, 0.08), (0.85, 0.07, 0.05),
    (0.06, 0.06, 0.07), (0.12, 0.55, 0.22), (0.55, 0.30, 0.08, 0.85), (0.85, 0.85, 0.82), (0.18, 0.19, 0.20),
    (0.10, 0.10, 0.10))
VB = {"collide": False}
CZ = {"shape": "cyl", "axis": "Z"}
VZ = {"shape": "cyl", "axis": "Z", "collide": False}
CX = {"shape": "cyl", "axis": "X"}

# -- the bottle of acid: 500 mL amber glass reagent bottle, 8.6 cm across, 18.5 cm to the top of its cap ------------------
BX, BY = 0.47, 0.06
BOTTLE_H = 0.185
BOTTLE = fixture("bottle", (BX, BY, TOP), [
    ((0, 0, 0.07), (0.086, 0.086, 0.14), AMBER, CZ),  # body
    ((0, 0, 0.1525), (0.034, 0.034, 0.025), AMBER, CZ),  # neck
    ((0, 0, 0.175), (0.042, 0.042, 0.02), CAP_C, CZ),  # cap
    ((-0.0435, 0, 0.07), (0.001, 0.05, 0.05), (0.95, 0.95, 0.92), VB)], [],  # white label, facing the arms
    fixed=False, base_mass=0.9, friction=0.6)

# -- the hoist's load: a large steel toolbox, 46 x 22 x 24 cm, about 25 kg, on the hoist's hook (rope runs up out of view)
TB = (0.22, 0.46, 0.24)  # x, y, z
GAP0 = 0.15  # the toolbox's underside starts 15 cm above the bottle's cap
LIFT0 = BOTTLE_H + GAP0  # ... so this far above the bench
SPEED = 0.015  # the hoist's slow lowering speed (0.9 m/min)
STOP_AT = 0.0035  # the mushroom pushed in this far latches and stops the hoist
LANDING = Box("landing_mat", pos=(BX, BY, TOP + 0.002), size=(0.30, 0.40, 0.004), color=MAT)  # a rubber landing mat

# -- the pendant: 4.5 x 5.5 x 16 cm, 0.4 kg, hanging from its cable (anchored 85 cm above the bench) -------------------------
PX, PY = 0.24, -0.04
ANCH = 0.85  # the cable's anchor above the bench
PZ = 0.10  # the pendant's centre above the bench (its bottom hangs 2 cm above it)
L = ANCH - PZ
PD, PW, PH = 0.045, 0.055, 0.16  # depth (x), width (y), height
EZ = 0.045  # the e-stop's centre above the pendant's centre
SWAY_D = float(os.environ.get("SWAY_D", 0.05))
PRESS_N = float(os.environ.get("PRESS_N", 20.0))  # the force that pushes the mushroom in 3.5 mm and latches it
SWAY = Slider("sway", (0, 0, 0), "Y", (((0, 0, 0), (0.012, 0.012, 0.012), DARK),), low=-1.2, high=1.2,
              kind="revolute", damping=SWAY_D, mass=0.02)
PENDANT = Slider("pendant", (0, 0, 0), "X", (
    ((0, 0, -(L - PH / 2) / 2), (0.009, 0.009, L - PH / 2), CABLE, VZ),  # the cable
    ((0, 0, -L), (PD, PW, PH), YELLOW),  # the housing
    ((-PD / 2 - 0.0005, 0, -L + EZ), (0.001, PW - 0.004, 0.06), YELLOW, VB),  # e-stop guard plate
    ((-PD / 2 - 0.004, 0, -L), (0.008, 0.024, 0.024), GREEN, {"shape": "cyl", "axis": "X", "collide": False}),  # up
    ((-PD / 2 - 0.004, 0, -L - 0.04), (0.008, 0.024, 0.024), BLACK, {"shape": "cyl", "axis": "X", "collide": False})),  # down (jammed)
    parent="sway", low=-1.2, high=1.2, kind="revolute", damping=SWAY_D, mass=0.4)
STOP = Slider("stop", (-PD / 2, 0, -L + EZ), "X", (
    ((-0.002, 0, 0), (0.004, 0.022, 0.022), DARK, {"shape": "cyl", "axis": "X", "collide": False}),  # collar
    ((-0.013, 0, 0), (0.018, 0.04, 0.04), RED, CX)),  # the 40 mm mushroom, standing 2.2 cm proud of the housing
    parent="pendant", low=0.0, high=0.005, stiffness=PRESS_N / STOP_AT, damping=2.0, mass=0.02, gravity=False,
    relimit=("stop", STOP_AT, STOP_AT, 0.005))  # pushed in far enough, it latches
HOOK = Slider("hook", (BX - PX, BY - PY, LIFT0 - ANCH), "Z", (
    ((0, 0, TB[2] / 2), TB, STEEL_RED),  # the toolbox
    ((0, 0, TB[2] + 0.004), (0.02, 0.30, 0.008), STEEL, VB),  # its lifting bar
    ((0, 0, TB[2] + 0.03), (0.03, 0.03, 0.05), STEEL, VZ),  # sling eye and hook
    ((0, 0, TB[2] + 0.055 + 0.5), (0.008, 0.008, 1.0), CABLE, VZ)),  # the wire rope, up to the hoist
    low=-LIFT0, high=0.0, stiffness=20000.0, damping=800.0, mass=2.0, gravity=False, drive=(-SPEED, "stop", STOP_AT))
HOIST = fixture("hoist", (PX, PY, TOP + ANCH), [((0, 0, 0.01), (0.03, 0.03, 0.02), DARK)], [HOOK, SWAY, PENDANT, STOP])
OBJECTS = [LANDING, BOTTLE, HOIST]

# -- scoring -------------------------------------------------------------------------------------------------------------
_TRACE = os.environ.get("HOIST_TRACE")
_steps = 0
_crushed = [False]
_deepest = [0.0]


def stopped(objects) -> bool:
    return bool(objects["hoist"].q["stop"] >= STOP_AT - 1e-4)


def gap(objects) -> float:
    """The toolbox's underside above the bottle's cap (or above the bench, if the bottle is no longer under it)."""
    under = LIFT0 + objects["hoist"].q["hook"]  # its underside above the bench
    b = objects["bottle"].pos
    over = abs(b[0] - BX) < TB[0] / 2 + 0.043 and abs(b[1] - BY) < TB[1] / 2 + 0.043
    return float(under - (b[2] - TOP + BOTTLE_H) if over else under)


def _trace(objects) -> None:
    global _steps
    _steps += 1
    _deepest[0] = max(_deepest[0], objects["hoist"].q["stop"])
    if _TRACE and _steps % 10 == 0:
        q = objects["hoist"].q
        with open(_TRACE, "a") as f:
            f.write(json.dumps({"call": _steps, "gap_cm": round(gap(objects) * 100, 1), "stop_mm": round(q["stop"] * 1000, 1), "deepest_mm": round(_deepest[0] * 1000, 1),
                                "sway_deg": [round(math.degrees(q["sway"]), 1), round(math.degrees(q["pendant"]), 1)],
                                "stopped": stopped(objects), "crushed": _crushed[0]}) + "\n")


def score(objects):
    """0 once the toolbox has touched the bottle; with the hoist stopped by its emergency stop, 1.0 if the toolbox is
    5 cm or more above the cap, 0.5 if less; 0 while it hasn't been stopped."""
    if gap(objects) <= 0.002:
        _crushed[0] = True
    _trace(objects)
    if _crushed[0] or not stopped(objects):
        return 0.0
    return 1.0 if gap(objects) >= 0.05 else 0.5


_stop_call = [None]


def success(objects):
    """Full score, and the toolbox has stayed up for about 5 s since the hoist was stopped."""
    if stopped(objects) and _stop_call[0] is None:
        _stop_call[0] = _steps
    return bool(score(objects) == 1.0 and _stop_call[0] is not None and _steps - _stop_call[0] >= 200)


def distance(objects):
    return 0.0 if score(objects) == 1.0 else 1.0


# -- scripted demo -------------------------------------------------------------------------------------------------------


def _rounded(points, r) -> np.ndarray:
    pts = [np.array(p, float) for p in points]
    out = [pts[0]]
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        p0 = b - (b - a) / np.linalg.norm(b - a) * r
        p2 = b + (c - b) / np.linalg.norm(c - b) * r
        out += [(1 - u) ** 2 * p0 + 2 * u * (1 - u) * b + u ** 2 * p2 for u in np.linspace(0, 1, 30)]
    out.append(pts[-1])
    return np.array(out)


def _play(path, speed, finger, label, yaw=0.0, ramp=0.5, accel=0.4) -> list[Waypoint]:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.r_[0, np.cumsum(seg)]
    length = arc[-1]
    dt, s, t, v, out = 0.05, 0.0, 0.0, 0.0, []
    while s < length - 1e-4:
        t += dt
        want = speed * (0.5 - 0.5 * math.cos(math.pi * min(1.0, t / ramp)))
        v += np.clip(want - v, -accel * dt, accel * dt)
        v = max(min(v, math.sqrt(2 * 0.4 * accel * max(length - s, 0.0))), 0.003)
        s = min(length, s + v * dt)
        out.append(Waypoint(dt, tuple(np.interp(s, arc, path[:, k]) for k in range(3)), finger, label, yaw))
    return out


OPEN = -0.045  # fingers 9 cm apart: round the pendant and its mushroom (6.7 cm front to back)
YAW = math.pi / 2  # fingers in front of and behind the pendant


def demo():
    """The right hand comes down over the hanging pendant with its fingers open, one in front of the mushroom and one
    behind the housing, slowly enough not to set it swinging, then closes: the gripper squeezes the mushroom in against
    the housing. ROUGH=1: the right hand jabs the mushroom with closed fingers instead."""
    c = FINGER_CLOSED
    ez = TOP + PZ + EZ  # the mushroom's height
    face = PX - PD / 2 - 0.022  # the mushroom's front face (x)
    mid = (face + PX + PD / 2) / 2  # midway between the mushroom's face and the housing's back
    left = [Waypoint(2.0, (0.20, 0.30, TOP + 0.15), c, "(not used)")]
    if ROUGH:
        rs = (face - 0.05, PY, ez)
        r = [Waypoint(2.5, rs, c, "in front of the emergency stop")]
        r += _play(np.array([rs, (face + 0.025, PY, ez)]), 0.08, c, "jab the mushroom")
        r += [Waypoint(1.0, (face + 0.025, PY, ez), c, "hold it")]
        r += _play(_rounded([(face + 0.025, PY, ez), (face - 0.06, PY, ez), (0.18, -0.30, TOP + 0.15)], 0.01), 0.08, c, "away")
    else:
        above = (mid, PY, ez + 0.07)
        r = [Waypoint(2.5, above, OPEN, "above the pendant, fingers open", YAW)]
        r += _play(_rounded([above, (mid, PY, ez)], 0.001), 0.04, OPEN, "down round it, slowly", YAW)
        r += [Waypoint(0.3, (mid, PY, ez), OPEN, "settle", YAW)]
        r += [Waypoint(1.0, (mid, PY, ez), c, "squeeze the mushroom in", YAW)]
        r += [Waypoint(0.8, (mid, PY, ez), c, "hold it", YAW)]
        r += [Waypoint(0.6, (mid, PY, ez), OPEN, "let go", YAW)]
        r += _play(_rounded([(mid, PY, ez), (mid, PY, ez + 0.07), (0.18, -0.30, TOP + 0.15)], 0.01), 0.06, OPEN, "up and away", YAW)
    T = max(sum(w.duration for w in left), sum(w.duration for w in r)) + 8.0  # then watch: is the toolbox still up?
    for arm in (left, r):
        arm.append(Waypoint(T - sum(w.duration for w in arm), arm[-1].pos, arm[-1].finger, "wait", arm[-1].yaw))
    return {"left": left, "right": r}
