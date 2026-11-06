"""On the station table, 30 cm out from the arm bases and facing them: a small white sliding window, 52 cm wide and
20 cm tall, fixed to the table. Its left half is a sliding sash that opens by sliding right, in front of the fixed
right pane. The sash slides stiffly (heavily damped). A slim chrome D-handle stands out from the sash's right edge; it
is mounted on a stiff spring that stands in for its fixings, and if it is pulled with more than about 14 N it tears off.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "The window is stiff and its handle is delicate. Grip the handle and slide the window open." \\
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
from slidekit import Slider, fixture  # noqa: E402  (importing slidekit hooks yam_arena so the window is simulated)
from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

WHITE, SHADE, CHROME = (0.93, 0.93, 0.91), (0.80, 0.81, 0.80), (0.72, 0.74, 0.77)
GLASS = (0.70, 0.85, 0.92, 0.30)  # sRGBA: see-through

# -- the window frame: fixed to the table, its frame at (X, 0, table top) ------------------------------------------------

X = 0.30  # the frame's centre plane, forward of the arm bases
Y_L, Y_R = 0.20, -0.30  # inner faces of the left and right jambs
H = 0.18  # the opening's height (sill to head)
SILL = HEAD = JAMB = 0.012
DEPTH = 0.028  # the frame's depth (x)
M = 0.014  # width of a sash or pane member
PANE_X = 0.007  # the fixed pane sits behind the sash
MEET_Y = -0.03  # the fixed pane's left edge, where the closed sash meets it
_MID, _SPAN = (Y_L + Y_R) / 2, Y_L - Y_R

FRAME = [
    ((0, _MID, SILL / 2), (DEPTH, _SPAN + 2 * JAMB, SILL), WHITE),
    ((0, _MID, SILL + H + HEAD / 2), (DEPTH, _SPAN + 2 * JAMB, HEAD), WHITE),
    ((0, Y_L + JAMB / 2, SILL + H / 2), (DEPTH, JAMB, H), WHITE),
    ((0, Y_R - JAMB / 2, SILL + H / 2), (DEPTH, JAMB, H), WHITE),
    ((PANE_X, MEET_Y, SILL + H / 2), (0.012, M, H), SHADE),  # the fixed pane: two stiles and the glass
    ((PANE_X, Y_R + M / 2, SILL + H / 2), (0.012, M, H), SHADE),
    ((PANE_X, (MEET_Y + Y_R) / 2, SILL + H / 2), (0.003, MEET_Y - Y_R, H), GLASS),
]

# -- the sash: slides along -y (to the right) to open -------------------------------------------------------------------

SASH_W = Y_L - MEET_Y  # 23 cm: closed, it fills the left half
SASH_D = 0.012  # its depth (x)
SASH_FRONT = -0.013  # its front face, relative to the frame's centre plane
SASH_Y = Y_L - SASH_W / 2  # its centre when closed
SASH_TRAVEL = 0.24
SASH_DAMPING = 300.0  # N s/m: pulled at 2 cm/s it drags with 6 N, at 10 cm/s with 30 N
SASH_MASS = 0.5

SASH = Slider("sash", (SASH_FRONT + SASH_D / 2, SASH_Y, SILL), "Y", (
    ((0, SASH_W / 2 - M / 2, H / 2), (SASH_D, M, H), WHITE),
    ((0, -(SASH_W / 2 - M / 2), H / 2), (SASH_D, M, H), WHITE),
    ((0, 0, M / 2), (SASH_D, SASH_W - 2 * M, M), WHITE),
    ((0, 0, H - M / 2), (SASH_D, SASH_W - 2 * M, M), WHITE),
    ((0, 0, H / 2), (0.003, SASH_W - 2 * M, H - 2 * M), GLASS),
), low=-SASH_TRAVEL, high=0.0, damping=SASH_DAMPING, friction=0.0, mass=SASH_MASS)

# -- the handle: a D-handle on the sash's right stile, on a spring that stands in for its fixings ------------------------

STAND = 0.032  # how far the grip bar stands out from the sash's face
BAR_X, BAR_Y, BAR_H = 0.010, 0.012, 0.070  # the grip bar's section and height
HANDLE_Y = -(SASH_W / 2 - M / 2)  # on the right stile, relative to the sash's centre
HANDLE_Z = 0.08  # the bar's centre above the sash's bottom
HANDLE_K = 800.0  # N/m
TEAR_AT = 0.018  # m of give: 800 N/m x 18 mm = 14.4 N
_FACE = -SASH_D / 2  # the sash's front face in its own frame

HANDLE = Slider("handle", (0, HANDLE_Y, HANDLE_Z), "Y", (
    ((_FACE - STAND - BAR_X / 2, 0, 0), (BAR_X, BAR_Y, BAR_H), CHROME),
    ((_FACE - STAND / 2, 0, BAR_H / 2 - 0.006), (STAND, 0.008, 0.008), CHROME),
    ((_FACE - STAND / 2, 0, -BAR_H / 2 + 0.006), (STAND, 0.008, 0.008), CHROME),
), low=-0.04, high=0.0, parent="sash", stiffness=HANDLE_K, damping=5.0, mass=0.03, breaks_at=TEAR_AT,
    broken_low=-0.40)

WINDOW = fixture("window", (X, 0.0, TOP), FRAME, [SASH, HANDLE])
OBJECTS = [WINDOW]

# -- scoring -----------------------------------------------------------------------------------------------------------

GOAL = 0.15  # m the sash has to be open
LEVELS = ((GOAL, 1.0), (0.12, 0.75), (0.08, 0.5), (0.04, 0.25))
STILL = 0.01  # m/s

_TRACE = os.environ.get("WINDOW_TRACE")  # optional: a file to log the window's state to every 10 steps, for debugging
_steps = 0


def opened(objects) -> float:
    """Metres the sash has slid open from closed."""
    return -objects["window"].q["sash"]


def torn(objects) -> bool:
    """The handle has torn off: it gave 18 mm (about 14 N) at some point in the trial."""
    return bool(objects["window"].broken["handle"])


def _trace(objects) -> None:
    global _steps
    _steps += 1
    if _TRACE and _steps % 10 == 0:
        w = objects["window"]
        with open(_TRACE, "a") as f:
            f.write(json.dumps({"call": _steps, "opened_cm": round(opened(objects) * 100, 2),
                                "give_mm": round(-w.q["handle"] * 1000, 1), "torn": torn(objects),
                                "speed_cm_s": round(-w.speed["sash"] * 100, 2)}) + "\n")


def score(objects):
    """0 if the handle has torn off. Otherwise 1.0 with the sash open 15 cm or more, 0.75 from 12 cm, 0.5 from 8 cm,
    0.25 from 4 cm, 0 below."""
    _trace(objects)
    if torn(objects):
        return 0.0
    d = opened(objects)
    return next((s for at, s in LEVELS if d >= at), 0.0)


def success(objects):
    """Open 15 cm or more with the handle on, and the sash still (held for SUCCESS_HOLD_S)."""
    return bool(score(objects) == 1.0 and abs(objects["window"].speed["sash"]) < STILL)


def distance(objects):
    """Metres the sash still has to slide to be 15 cm open (0 once it is)."""
    return float(max(0.0, GOAL - opened(objects)))


# -- scripted demo -----------------------------------------------------------------------------------------------------

PULL_S = 10.0  # the pull's length: about 2 cm/s, so the handle carries about 6 N
OPEN = 0.20  # how far the demo opens the sash (the goal is 15 cm)
MOVE = 0.08  # m/s, moving in free space
CORNER = 0.025  # corner radius on the way in (m)
TURN = 0.02  # corner radius from the pull into the move away
EASE = 0.015  # the hand opens over the last 1.5 cm of the pull, still moving
OPEN_FINGERS = -0.018


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


def _play(path, speed, finger, label, ramp=0.8) -> list[Waypoint]:
    """Follow ``path`` with one speed-up, one slow-down and no stop between, one waypoint per control step.
    ``speed(s)`` and ``finger(s)`` depend on the distance ``s`` along the path."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.r_[0, np.cumsum(seg)]
    length = arc[-1]
    dt, s, t, v, out = 0.05, 0.0, 0.0, 0.0, []
    while s < length - 1e-4:
        t += dt
        want = speed(s) * (0.5 - 0.5 * math.cos(math.pi * min(1.0, t / ramp)))
        v += np.clip(want - v, -0.15 * dt, 0.15 * dt)  # never change speed abruptly
        v = max(min(v, math.sqrt(2 * 0.06 * max(length - s, 0.0))), 0.003)  # slow down to stop at the end
        s = min(length, s + v * dt)
        out.append(Waypoint(dt, tuple(np.interp(s, arc, path[:, k]) for k in range(3)), finger(s), label))
    return out


def demo():
    """The right arm reaches round to the handle from the room side, grips it, pulls the sash 20 cm open slowly in one
    continuous motion, lets go while still moving and draws back."""
    gx = X + SASH_FRONT + SASH_D / 2 + _FACE - STAND - BAR_X / 2 - 0.003  # the grip bar's centre, 3 mm short
    y = SASH_Y + HANDLE_Y
    z = TOP + SILL + HANDLE_Z
    end = y - OPEN - 0.012
    high = (0.15, y - 0.10, z + 0.10)  # room side of the frame and above the handle: the fingers never cross the frame
    before, grip, pulled, away = (gx - 0.07, y, z), (gx, y, z), (gx, end, z), (gx - 0.09, end, z)
    pull_v = (y - end) / PULL_S
    corner = (y - end) - TURN

    def pull_speed(s):
        return pull_v if s < corner else pull_v + (MOVE - pull_v) * min(1.0, (s - corner) / 0.03)

    def fingers(s):
        if s < corner - EASE:
            return FINGER_CLOSED
        if s >= corner:
            return OPEN_FINGERS
        return FINGER_CLOSED + (OPEN_FINGERS - FINGER_CLOSED) * (s - corner + EASE) / EASE

    right = [Waypoint(3.0, high, OPEN_FINGERS, "from ready, back and up")]  # the wrist can only turn in this first move
    right += _play(_rounded([high, before, grip], CORNER), lambda s: MOVE, lambda s: OPEN_FINGERS,
                   "reach around the handle")
    right += [Waypoint(0.5, grip, FINGER_CLOSED, "grip")]
    right += _play(_rounded([grip, pulled, away], TURN), pull_speed, fingers, "pull, let go and draw back")
    return {"right": right}
