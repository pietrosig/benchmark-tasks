"""On the station table, 30 cm out from the arm bases: an old lighting panel, a wooden board fixed to the table with four
brass toggle switches in a row and a lamp in front of each. Each toggle is a slim brass lever on a pivot that clicks
over to either side: leaning away from the arms is on, leaning towards them is off. Three lamps are lit; the third one
from the right is already off. Pulled just past the middle, a toggle clicks over by itself and its lamp goes out. The
levers are old and brittle: forced on past their end stop they snap off at the base.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "The panel's switches are old and brittle. Turn off the lights." \\
           --scorer task_score -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 ...
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from togglekit import Slider, fixture, start_at  # noqa: E402  (importing togglekit hooks yam_arena so the panel is simulated)
from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

WOOD, WOOD2, BRASS, BLACK, BULB, SHADE, PLATE = ((0.45, 0.28, 0.16), (0.36, 0.22, 0.12), (0.85, 0.68, 0.30),
                                                 (0.08, 0.08, 0.09), (1.0, 0.86, 0.30), (0.22, 0.22, 0.24),
                                                 (0.72, 0.72, 0.70))
VB = {"collide": False}
VZ = {"shape": "cyl", "axis": "Z", "collide": False}
CZ = {"shape": "cyl", "axis": "Z"}

# -- the panel: a board fixed to the table, centred at (X, 0, table top) ------------------------------------------------

X = 0.30
BOARD = (0.09, 0.34, 0.02)  # depth (x), length (y), thickness
BT = BOARD[2]
YS = (-0.12, -0.04, 0.04, 0.12)  # the four toggles along the row (y), right to left
OFF_AT_START = (2,)  # the third from the right is already off
PIVOT_X = 0.004  # the toggles' pivots, relative to the board's centre
PIVOT_Z = BT + 0.003
LAMP_X = -0.032  # the lamps sit in front of the toggles (towards the arms)

FIXED = [((0, 0, BT / 2), BOARD, WOOD), ((0, 0, BT + 0.0004), (BOARD[0] - 0.01, BOARD[1] - 0.01, 0.0008), WOOD2, VB)]
for _y in YS:
    FIXED += [((PIVOT_X, _y, BT + 0.001), (0.03, 0.022, 0.002), PLATE, VB),  # the switch plate
              ((LAMP_X, _y, BT + 0.006), (0.016, 0.016, 0.012), BULB, CZ)]   # the lamp (lit)

# -- the toggles ----------------------------------------------------------------------------------------------------------

THROW = math.radians(20)  # each lever leans this far either way: + (away from the arms) is on, - is off
CLICK = -0.03  # rad: pulled this far past the middle, a toggle clicks over to off by itself
TOGGLE_K = 0.02  # N m/rad: the click spring (about 0.25 N at the lever's tip)
LEVER = 0.035  # the lever's length above its boss
LEVER_D, KNOB_D = 0.004, 0.007
BEND_K = 2.0  # N m/rad: how stiff the lever is at its base
SNAP_AT = math.radians(3.0)  # it snaps once bent this far: about 0.105 N m, some 3 N at the knob
TIP_R = 0.004 + LEVER  # from the pivot to the knob's centre

PARTS = []
for _i, _y in enumerate(YS):
    _off = _i in OFF_AT_START
    # the toggle: a black boss on a pivot (rotation about +y: positive leans the lever away from the arms)
    PARTS.append(Slider(f"toggle{_i}", (PIVOT_X, _y, PIVOT_Z), "Y", (((0, 0, 0.003), (0.012, 0.012, 0.006), BLACK, VZ),),
                        low=-THROW, high=THROW, kind="revolute", stiffness=TOGGLE_K, damping=0.0005, mass=0.004,
                        target=-THROW if _off else THROW,
                        retarget=None if _off else (f"toggle{_i}", CLICK, -THROW)))
    start_at("panel", f"toggle{_i}", -THROW if _off else THROW)
    # the brass lever, on a stiff spring at its base that stands in for the brittle metal: it snaps at SNAP_AT
    PARTS.append(Slider(f"lever{_i}", (0, 0, 0.004), "Y",
                        (((0, 0, LEVER / 2), (LEVER_D, LEVER_D, LEVER), BRASS, CZ), ((0, 0, LEVER), (KNOB_D, KNOB_D, KNOB_D), BRASS, CZ)),
                        parent=f"toggle{_i}", low=-1.4, high=1.4, kind="revolute", stiffness=BEND_K, damping=0.02,
                        mass=0.004, breaks_at=SNAP_AT, limp_damping=0.002))
    # the lamp going out: a dark shade over the bulb, hidden under the table while the lamp is lit
    PARTS.append(Slider(f"shade{_i}", (LAMP_X, _y, BT + 0.0065), "Z", (((0, 0, 0), (0.018, 0.018, 0.014), SHADE, VZ),),
                        low=-0.3, high=0.0, stiffness=3000.0, damping=30.0, mass=0.01, gravity=False,
                        target=-0.125, follows=f"toggle{_i}", ratio=-0.25 / (2 * THROW)))
    start_at("panel", f"shade{_i}", 0.0 if _off else -0.25)

PANEL = fixture("panel", (X, 0.0, TOP), FIXED, PARTS)
OBJECTS = [PANEL]

# -- scoring -----------------------------------------------------------------------------------------------------------

LIT_AT_START = tuple(i for i in range(len(YS)) if i not in OFF_AT_START)
OFF_BELOW = -0.6 * THROW  # a toggle leaning more than 12 deg towards the arms is off
STILL = 0.05  # rad/s

_TRACE = os.environ.get("LIGHTS_TRACE")  # optional: a file to log the panel's state to every 10 steps, for debugging
_steps = 0


def is_off(objects, i) -> bool:
    return bool(objects["panel"].q[f"toggle{i}"] <= OFF_BELOW)


def snapped(objects) -> list[int]:
    """The toggles whose lever has snapped (at any point in the trial; it stays snapped)."""
    return [i for i in range(len(YS)) if objects["panel"].broken[f"lever{i}"]]


def lit(objects) -> list[int]:
    return [i for i in range(len(YS)) if not is_off(objects, i)]


def _trace(objects) -> None:
    global _steps
    _steps += 1
    if _TRACE and _steps % 10 == 0:
        p = objects["panel"]
        with open(_TRACE, "a") as f:
            f.write(json.dumps({"call": _steps, "toggle_deg": [round(math.degrees(p.q[f"toggle{i}"]), 1) for i in range(len(YS))],
                                "bend_deg": [round(math.degrees(p.q[f"lever{i}"]), 2) for i in range(len(YS))],
                                "lit": lit(objects), "snapped": snapped(objects)}) + "\n")


def score(objects):
    """0 if any lever has snapped. Otherwise a third for each of the three lamps lit at the start that is now off."""
    _trace(objects)
    if snapped(objects):
        return 0.0
    return sum(is_off(objects, i) for i in LIT_AT_START) / len(LIT_AT_START)


def success(objects):
    """All lamps off, no lever snapped, and every toggle at rest (held for SUCCESS_HOLD_S)."""
    p = objects["panel"]
    return bool(score(objects) == 1.0 and all(abs(p.speed[f"toggle{i}"]) < STILL for i in range(len(YS))))


def distance(objects):
    """How many lamps are still lit (0 once all are off)."""
    return float(len(lit(objects)))


# -- scripted demo -----------------------------------------------------------------------------------------------------

V_MOVE = 0.025  # m/s, moving between toggles
V_PULL = 0.012  # m/s, pulling a lever
CORNER = 0.008  # corner radius of the path
BEHIND = 0.016  # the hand comes down this far behind the lever's knob (away from the arms)
PULL_TO = -0.12  # rad: the pull ends just past the click point; the toggle clicks the rest of the way by itself


def tip(i, q=THROW) -> np.ndarray:
    """World position of toggle i's knob at angle q."""
    return np.array([X + PIVOT_X + TIP_R * math.sin(q), YS[i], TOP + PIVOT_Z + TIP_R * math.cos(q)])


def _rounded(points, r) -> np.ndarray:
    """A dense path through ``points`` with each corner rounded ``r`` either side, so the hand never stops there."""
    pts = [np.array(p, float) for p in points]
    out = [pts[0]]
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        p0 = b - (b - a) / np.linalg.norm(b - a) * r
        p2 = b + (c - b) / np.linalg.norm(c - b) * r
        out += [(1 - u) ** 2 * p0 + 2 * u * (1 - u) * b + u ** 2 * p2 for u in np.linspace(0, 1, 30)]
    out.append(pts[-1])
    return np.array(out)


def _play(path, speed, label, ramp=1.0, accel=0.05) -> list[Waypoint]:
    """Follow ``path`` with one speed-up, one slow-down and no stop between, one waypoint per control step;
    ``speed(s)`` depends on the distance ``s`` along the path, and the speed never changes faster than ``accel``."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.r_[0, np.cumsum(seg)]
    length = arc[-1]
    dt, s, t, v, out = 0.05, 0.0, 0.0, 0.0, []
    while s < length - 1e-4:
        t += dt
        want = speed(s) * (0.5 - 0.5 * math.cos(math.pi * min(1.0, t / ramp)))
        v += np.clip(want - v, -accel * dt, accel * dt)
        v = max(min(v, math.sqrt(2 * 0.4 * accel * max(length - s, 0.0))), 0.003)  # slow down to stop at the end
        s = min(length, s + v * dt)
        out.append(Waypoint(dt, tuple(np.interp(s, arc, path[:, k]) for k in range(3)), FINGER_CLOSED, label))
    return out


def route(toggles, side) -> list[Waypoint]:
    """One continuous path for an arm: over each lit toggle, down just behind its knob, a slow pull just past the
    middle (it clicks over by itself), up, on to the next, and finally up and back a little."""
    pts = []
    for i in toggles:
        z = tip(i)[2] - 0.006  # just under the knob
        a = (tip(i)[0] + BEHIND, YS[i], z)
        b = (tip(i, PULL_TO)[0], YS[i], z)
        pts += [(a[0], a[1], z + 0.04), a, b, (b[0] + 0.008, b[1], z + 0.04)]
    pts.append((pts[-1][0] - 0.03, pts[-1][1] + side * 0.03, pts[-1][2] + 0.04))
    path = _rounded(pts, CORNER)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    z_pull = min(p[2] for p in pts)

    def speed(s):
        return V_PULL if np.interp(s, arc, path[:, 2]) < z_pull + 0.012 else V_MOVE

    return [Waypoint(4.0, tuple(path[0]), FINGER_CLOSED, "over the first lit toggle")] + _play(path, speed, "switch the lit ones off")


def demo():
    """Each arm switches off the lit toggles on its side, one at a time, in one continuous motion; the toggle that is
    already off is left alone."""
    right = route([i for i in LIT_AT_START if YS[i] < 0], -1)
    left = route([i for i in LIT_AT_START if YS[i] > 0], 1)
    end = max(sum(w.duration for w in right), sum(w.duration for w in left)) + 2.0
    for arm in (right, left):
        rest = end - sum(w.duration for w in arm)
        if rest > 1e-6:
            arm.append(Waypoint(rest, arm[-1].pos, FINGER_CLOSED, "wait"))
    return {"left": left, "right": right}
