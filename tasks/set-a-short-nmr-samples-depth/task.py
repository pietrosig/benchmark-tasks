"""On the station table: an NMR sample tube standing in a small rack on the right, a spinner turbine seated in a depth
gauge in front of the right arm, and on the left a segment of a sample changer's carousel: three seat columns in a row,
the outer two holding spinners with their tubes, the middle one open.

The tube (5 mm glass, 178 mm long, blue cap) holds a short sample: a 30 mm orange column at its bottom. Its top 50 mm
is sleeved dark green (the place to hold it); a red band 80 mm up marks the part below, which goes into the probe and
must not be touched. The spinner's bore grips a tube with a few newtons of friction. The gauge's front window has a mm
scale and a black line: a short sample is set by centring its column on that line (the gauge's floor is 20 mm below the
line, so pushing the tube to the floor leaves the column 4 mm low). The carousel's open position has the same window,
scale and line, and no floor near the tube's end: the tube hangs from its spinner.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --instruction "$(cat instruction.txt)" \\
           --scorer task_score -E object_jitter=0 -E control_hz=20 ...
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from nmrkit import Slider, fixture  # noqa: E402  (importing nmrkit hooks yam_arena so the fixtures are simulated)
from yam_arena import demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"


def _env(name, default):
    return float(os.environ.get(name, default))


WHITE, GREY, DARK, BLACK = (0.92, 0.92, 0.90), (0.62, 0.63, 0.64), (0.18, 0.18, 0.20), (0.03, 0.03, 0.03)
POM = (0.95, 0.95, 0.93)  # the spinner's white plastic
BRASS = (0.78, 0.62, 0.30)
GLASS = (0.80, 0.90, 0.95, 0.35)  # see-through
SAMPLE = (0.95, 0.45, 0.05)
SLEEVE = (0.10, 0.35, 0.20)
TAPE = (0.85, 0.08, 0.08)
CAP = (0.10, 0.30, 0.85)
NOCOL = {"collide": False}
CYL = {"shape": "cyl"}
CYL_V = {"shape": "cyl", "collide": False}

# -- dimensions (m); heights are above the bench top ----------------------------------------------------------------------

SEAT = 0.093  # the spinner's shoulder rests on the gauge (and on a carousel position) this high
D_LINE = 0.062  # the gauge's line is this far below the shoulder: the centre of the probe's detected region (estimate)
LINE = SEAT - D_LINE  # 31 mm above the bench
FLOOR = LINE - 0.020  # the gauge's floor stop, 20 mm below the line (the Bruker setting for newer probes)
TUBE_L = 0.178  # 7 inch tube
TUBE_R = 0.0025  # 5 mm outside diameter
COLUMN = (0.001, 0.031)  # the sample column, from 1 mm above the tube's bottom (its glass bottom) to 31 mm
COL_MID = sum(COLUMN) / 2  # 16 mm above the tube's bottom
GRIP_ZONE = (0.128, 0.178)  # the sleeved top 50 mm
NO_TOUCH = 0.080  # the lower 80 mm: below the red band
TURB_H, TURB_R = 0.018, 0.012  # the spinner's turbine (above the shoulder)
STEM_H, STEM_R = 0.015, 0.007  # its stem (below the shoulder, into the seat)
ROLL_Z = 0.009  # the bore's sprung pads, this high above the shoulder

GAUGE_XY = (0.27, -0.07)
RACK_XY = (0.22, -0.20)
SLOT_XY = (0.30, 0.16)  # the changer rack's open position
PITCH = 0.055  # its positions are 55 mm apart, in a row across the left arm's reach
_d = np.array([SLOT_XY[0], SLOT_XY[1] - 0.305])
_d /= np.linalg.norm(_d)
ROW = np.array([_d[1], -_d[0]])  # the row's direction (towards the right arm's side)
NB_XY = {"nb_right": tuple(float(v) for v in np.array(SLOT_XY) + PITCH * ROW),
         "nb_left": tuple(float(v) for v in np.array(SLOT_XY) - PITCH * ROW)}


def _box(c, s, k, o=None):
    return (tuple(c), tuple(s), k, o or {})


def _square_tube(z0, z1, inner, outer, color, o=None, open_front=False):
    """Four walls round a square hole of half-width ``inner`` (outer half-width ``outer``), from z0 to z1 (local)."""
    h, zc, w = z1 - z0, (z0 + z1) / 2, outer - inner
    walls = [
        _box((0, inner + w / 2, zc), (2 * outer, w, h), color, o),
        _box((0, -(inner + w / 2), zc), (2 * outer, w, h), color, o),
        _box((inner + w / 2, 0, zc), (w, 2 * inner, h), color, o),
    ]
    if not open_front:
        walls.append(_box((-(inner + w / 2), 0, zc), (w, 2 * inner, h), color, o))
    return walls


def _ring(z0, z1, r_in, apothem, color, n=12):
    """A ring of ``n`` flat walls (a polygon with this apothem outside, a bore of radius about ``r_in``)."""
    out = []
    w = 2 * apothem * math.tan(math.pi / n) + 0.0002
    for k in range(n):
        a = 2 * math.pi * k / n
        rc = (apothem + r_in) / 2
        out.append(_box((rc * math.cos(a), rc * math.sin(a), (z0 + z1) / 2), (apothem - r_in, w, z1 - z0), color,
                        {"rz": math.degrees(a)}))
    return out


COL_W = 0.0095  # every seat column is 19 mm square: narrower than the turbine, so fingers can close round a spinner on it


def _column(foot, window, color=WHITE):
    """An upright 19 mm square column on a square foot (``foot`` = (side, thickness)): a 12 mm bore, the stem's 15 mm
    seat in its top 16 mm, and (``window``) a clear front window with a mm scale and the black line."""
    side, t = foot
    out = [_box((0, 0, t / 2), (side, side, t), GREY)]
    out += _square_tube(t, SEAT - 0.016, 0.006, COL_W, color, open_front=window)
    out += _ring(SEAT - 0.016, SEAT, 0.0075, COL_W, color)  # the seat: a 12-sided collar, 19 mm across its flats
    if window:
        out.append(_box((-0.00775, 0, (t + SEAT - 0.016) / 2), (0.0035, 0.012, SEAT - 0.016 - t), GLASS))
        for i in range(-4, 9):  # ticks every 5 mm, longer every 10 mm, beside the window
            z = LINE + 0.005 * i
            if t + 0.002 < z < SEAT - 0.018:
                out.append(_box((-COL_W - 0.0002, 0.0078, z), (0.0006, 0.003 if i % 2 == 0 else 0.0018, 0.0007), BLACK,
                                NOCOL))
        out.append(_box((-COL_W - 0.0003, 0, LINE), (0.0008, 2 * COL_W + 0.001, 0.0014), BLACK, NOCOL))  # the line
    return out


# -- the depth gauge (fixed): a column with a window and scale at the front, a floor stop, the spinner's seat on top ----

_G = _column((0.070, 0.008), True)
_G.append(_box((0, 0, FLOOR - 0.0015), (0.012, 0.012, 0.003), DARK))  # the floor stop, 20 mm below the line

# -- the spinner (free): a turbine on a stem, its bore gripping the tube through four sprung pads -------------------------

_S = [
    _box((0, 0, TURB_H / 2), (2 * TURB_R, 2 * TURB_R, TURB_H), POM, CYL_V),
    _box((0, 0, -STEM_H / 2), (2 * STEM_R, 2 * STEM_R, STEM_H), POM, CYL_V),
    _box((0, 0, TURB_H - 0.002), (2 * TURB_R + 0.0004, 2 * TURB_R + 0.0004, 0.002), DARK, CYL_V),  # its top band
]
_S += _ring(0.0, TURB_H, 0.0045, 0.0115, POM)  # the turbine: a 12-sided ring, 23 mm across its flats
_S += _ring(-STEM_H, 0.0, 0.0042, STEM_R - 0.0002, POM)  # the stem round the bore
PAD_K = _env("PAD_K", 500.0)  # N/m, each pad's spring
PAD_PRE = 0.0012  # the spring's target is 1.2 mm inside the pad's stop: a preload of 0.6 N on its own
PAD_R = 0.002  # each pad is a roller 4 mm across, its axis tangential, so a tube end pushes it aside
PAD_REST = TUBE_R + PAD_R - 0.0004  # the rollers' centres from the axis at rest: they overlap a tube by 0.4 mm
PAD_Z = (0.013, -0.008)  # two rings of four pads, in the turbine and in the stem: the bore holds the tube upright


def _pad(name, axis, sign, z):
    i = "XY".index(axis)
    origin = [0.0, 0.0, z]
    origin[i] = sign * PAD_REST
    size = [0.0, 0.0, 2 * PAD_R]
    size[i] = 2 * PAD_R
    size[1 - i] = 0.006
    roller = _box((0, 0, 0), tuple(size), (0.25, 0.25, 0.28), {"shape": "cyl", "axis": "XY"[1 - i]})
    low, high = (0.0, 0.0015) if sign > 0 else (-0.0015, 0.0)
    return Slider(name, tuple(origin), axis, (roller,), low=low, high=high, stiffness=PAD_K, damping=0.5,
                  mass=0.004, gravity=False, target=-sign * PAD_PRE)


PADS = [_pad(f"pad{k}_{a.lower()}{'p' if g > 0 else 'n'}", a, g, z) for k, z in enumerate(PAD_Z)
        for a in "XY" for g in (1, -1)]

# -- the tube (free): glass, a cap, the sample column, the sleeve and the red band (the last three drawn only) -----------

TUBE_MASS = _env("TUBE_MASS", 0.020)  # kg (a real one is about 6 g with cap and sample; heavier for steady contacts)


def _tube_boxes(c=GLASS):
    return [
        _box((0, 0, 0.001), (2 * TUBE_R - 0.0012, 2 * TUBE_R - 0.0012, 0.002), c, CYL),  # the rounded end, as a step
        _box((0, 0, 0.002 + (TUBE_L - 0.002) / 2), (2 * TUBE_R, 2 * TUBE_R, TUBE_L - 0.002), c, CYL),
        _box((0, 0, TUBE_L + 0.006), (0.0068, 0.0068, 0.012), CAP, CYL),
        _box((0, 0, sum(COLUMN) / 2), (0.0036, 0.0036, COLUMN[1] - COLUMN[0]), SAMPLE, CYL_V),
        _box((0, 0, sum(GRIP_ZONE) / 2), (2 * TUBE_R + 0.0005, 2 * TUBE_R + 0.0005, GRIP_ZONE[1] - GRIP_ZONE[0]),
             SLEEVE, CYL_V),
        _box((0, 0, NO_TOUCH - 0.002), (2 * TUBE_R + 0.0005, 2 * TUBE_R + 0.0005, 0.004), TAPE, CYL_V),
    ]


# -- the rack (fixed): a block with one 7 mm hole, the tube standing in it --------------------------------------------

_R = _square_tube(0.0, 0.040, 0.0035, 0.015, GREY)

# -- the carousel segment (fixed): three seat columns like the gauge's in a row, the middle one open ------------------

_C = []
for xy in (SLOT_XY, *NB_XY.values()):  # three columns like the gauge's, but no floor near the tube's end
    off = (xy[0] - SLOT_XY[0], xy[1] - SLOT_XY[1], 0.0)
    for c, sz, k, o in _column((0.032, 0.006), xy == SLOT_XY, color=(0.80, 0.82, 0.86)):
        _C.append(_box((float(c[0] + off[0]), float(c[1] + off[1]), float(c[2])), sz, k, o))

# -- a neighbouring position's spinner with its tube, as one loose rigid piece ------------------------------------------

NB_DEPTH = D_LINE + COL_MID  # their tubes' bottoms below the shoulder (set right)


def _neighbour():
    out = [b for b in _S]
    for c, s, k, o in _tube_boxes():
        out.append(_box((c[0], c[1], c[2] - NB_DEPTH), s, k, o))
    return out


# -- the objects ----------------------------------------------------------------------------------------------------

GAUGE = fixture("gauge", (*GAUGE_XY, TOP), _G, [])
SPINNER = fixture("spinner", (*GAUGE_XY, TOP + SEAT + 0.0003), _S, PADS, fixed=False, base_mass=0.010, friction=0.5,
                  contacts=True)
TUBE = fixture("tube", (*RACK_XY, TOP + 0.0005), _tube_boxes(), [], fixed=False, base_mass=TUBE_MASS, friction=0.5,
               contacts=True)
RACK = fixture("rack", (*RACK_XY, TOP), _R, [])
CHANGER = fixture("changer", (*SLOT_XY, TOP), _C, [])
NB_RIGHT = fixture("nb_right", (*NB_XY["nb_right"], TOP + SEAT + 0.0003), _neighbour(), [], fixed=False,
                   base_mass=0.030, friction=0.5)
NB_LEFT = fixture("nb_left", (*NB_XY["nb_left"], TOP + SEAT + 0.0003), _neighbour(), [], fixed=False,
                  base_mass=0.030, friction=0.5)
OBJECTS = [GAUGE, RACK, CHANGER, SPINNER, TUBE, NB_RIGHT, NB_LEFT]

# -- state helpers ------------------------------------------------------------------------------------------------------


def _rot(q_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in q_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def axis_of(o) -> np.ndarray:
    return _rot(o.quat_xyzw)[:, 2]


def tilt_deg(o) -> float:
    return math.degrees(math.acos(float(np.clip(axis_of(o)[2], -1.0, 1.0))))


def bottom(o) -> np.ndarray:
    """The tube's bottom centre (its origin), height above the bench top in z."""
    p = np.asarray(o.pos, float).copy()
    p[2] -= TOP
    return p


def shoulder(o) -> np.ndarray:
    """A spinner's shoulder centre (its origin), height above the bench top in z."""
    p = np.asarray(o.pos, float).copy()
    p[2] -= TOP
    return p


SEAT_XY_TOL, SEAT_Z_TOL, SEAT_TILT = 0.002, 0.0015, 3.0
SEATS = {"gauge": GAUGE_XY, "slot": SLOT_XY}


def seated_in(sp) -> str | None:
    """Where the spinner is seated: "gauge", "slot" or None (shoulder on the seat within 1.5 mm, axis within 2 mm, upright
    within 3 degrees)."""
    s = shoulder(sp)
    if abs(s[2] - SEAT) > SEAT_Z_TOL or tilt_deg(sp) > SEAT_TILT:
        return None
    for name, xy in SEATS.items():
        if math.hypot(s[0] - xy[0], s[1] - xy[1]) <= SEAT_XY_TOL:
            return name
    return None


def depth_error(sp, tb) -> float:
    """How far (m) the sample column's centre is from the line's depth below the spinner's shoulder, along the spinner's
    axis: positive too low. Read on the gauge's or the open position's mm scale in a real trial."""
    a = axis_of(sp)
    col = bottom(tb) + axis_of(tb) * COL_MID
    return float(-(col - shoulder(sp)) @ a - D_LINE)


def through(sp, tb) -> bool:
    """The tube passes through the spinner's bore: within 1 mm of its axis at the shoulder, bottom below the stem, top
    above the turbine, and parallel within 3 degrees."""
    a, s, b = axis_of(sp), shoulder(sp), bottom(tb)
    t = axis_of(tb)
    rel = s - b
    along = float(rel @ t)
    off = np.linalg.norm(rel - along * t)
    ang = math.degrees(math.acos(float(np.clip(a @ t, -1, 1))))
    return bool(off < 0.001 and along > STEM_H and TUBE_L - along > TURB_H and ang < 3.0)


GRIP_LINKS = ("gripper", "tip_left", "tip_right")
CLEAR = 0.04  # at the end both tool points are at least 4 cm from the tube's axis and from the spinner
SPIN_GRIP = (0.12, 0.45)  # finger opening (0 shut .. 1 open) closed on the 24 mm turbine
_ever = {"dropped": False, "low_touch": False, "by_tube": False, "knocked": False, "other_link": False}
_why: list = []
_steps = 0
_TRACE = os.environ.get("TASK_TRACE")


def reset_credit() -> None:
    global _steps, _unheld
    _unheld = 0
    for k in _ever:
        _ever[k] = False
    _why.clear()
    _steps = 0


def _latch(k, why):
    if not _ever[k]:
        _ever[k] = True
        _why.append(f"step {_steps}: {why}")


def _dist_to_axis(p, o, origin) -> float:
    a = axis_of(o)
    r = np.asarray(p, float) - (origin + np.array([0, 0, TOP]))
    return float(np.linalg.norm(r - (r @ a) * a))


def _holding_spinner(sp, hands) -> bool:
    """A gripper is closed on the spinner's turbine: its tool point at the turbine, its fingers at the turbine's width,
    and a finger or the gripper touching the spinner's body (PhysX contacts)."""
    s = shoulder(sp) + np.array([0, 0, TOP]) + axis_of(sp) * (TURB_H / 2)
    touching = {link.split("/", 1)[0] for link in (sp.contacts or {}).get("body", ())
                if link.split("/", 1)[1] in GRIP_LINKS}
    for arm, h in hands.items():
        d = np.asarray(h["tcp"], float) - s
        if (math.hypot(d[0], d[1]) < 0.012 and abs(d[2]) < 0.012 and SPIN_GRIP[0] <= h["open"] <= SPIN_GRIP[1]
                and arm in touching):
            return True
    return False


UNHELD_STEPS = 5  # the spinner off its seat with no gripper on it for this many control steps in a row (0.25 s)
_unheld = 0


def _watch(objects) -> None:
    sp, tb = objects["spinner"], objects["tube"]
    b = bottom(tb)
    # dropped: the tube tipped past 30 degrees, or its bottom near the bench away from the rack and the gauge's floor
    # and the carousel's open position; the spinner fell off (well below its seat, not on the tube)
    if tilt_deg(tb) > 30.0:
        _latch("dropped", f"tube tipped {tilt_deg(tb):.0f} deg")
    home = [RACK_XY, GAUGE_XY, SLOT_XY]
    if b[2] < 0.004 and min(math.hypot(b[0] - x, b[1] - y) for x, y in home) > 0.01:
        _latch("dropped", "tube bottom on the bench")
    s = shoulder(sp)
    if s[2] < SEAT - 0.03 or tilt_deg(sp) > 30.0:
        _latch("dropped", "spinner fell")
    # the lower 80 mm touched: a gripper link on the tube with its tool point within 95 mm of the tube's bottom
    c = tb.contacts or {}
    hands = tb.hands
    for link in c.get("body", ()):
        arm, name = link.split("/", 1)
        if name not in GRIP_LINKS:
            _latch("other_link", f"{link} touched the tube")
            continue
        tcp = np.asarray(hands[arm]["tcp"], float)
        up = float((tcp - (b + np.array([0, 0, TOP]))) @ axis_of(tb))
        if up < NO_TOUCH + 0.015:
            _latch("low_touch", f"{link} on the tube {up * 1000:.0f} mm above its bottom")
    # carried by the tube: the spinner off every seat and above it while no gripper holds its body
    global _unheld
    if seated_in(sp) is None and s[2] > SEAT + 0.003 and through(sp, tb) and not _holding_spinner(sp, hands):
        _unheld += 1
        if _unheld >= UNHELD_STEPS:
            _latch("by_tube", "spinner lifted without a gripper on it")
    else:
        _unheld = 0
    for n, (x, y) in NB_XY.items():
        p = shoulder(objects[n])
        if math.hypot(p[0] - x, p[1] - y) > 0.002 or abs(p[2] - SEAT) > 0.002 or tilt_deg(objects[n]) > 2.0:
            _latch("knocked", f"{n} moved")


def clear_of(objects) -> bool:
    sp, tb = objects["spinner"], objects["tube"]
    for h in tb.hands.values():
        p = np.asarray(h["tcp"], float)
        if _dist_to_axis(p, tb, bottom(tb)) < CLEAR or np.linalg.norm(p - (shoulder(sp) + [0, 0, TOP])) < CLEAR:
            return False
    return True


def evaluate(objects) -> tuple[float, str]:
    """(score, why). See ``score``."""
    sp, tb = objects["spinner"], objects["tube"]
    if any(_ever.values()):
        return 0.0, "; ".join(_why)
    where = seated_in(sp)
    if where is None or not through(sp, tb) or tilt_deg(tb) > 3.0:
        return 0.0, "tube not through a seated spinner"
    if not clear_of(objects):
        return 0.0, "a gripper is still within 4 cm"
    e = abs(depth_error(sp, tb))
    if e <= 0.001:
        return (1.0 if where == "slot" else 0.75), f"{where}, column {e * 1000:.1f} mm off the line"
    if e <= 0.003:
        return 0.5, f"{where}, column {e * 1000:.1f} mm off the line"
    return 0.25, f"{where}, column {e * 1000:.1f} mm off the line"


def _trace(objects, s, why) -> None:
    if not _TRACE or _steps % 10:
        return
    sp, tb = objects["spinner"], objects["tube"]
    with open(_TRACE, "a") as f:
        f.write(json.dumps({
            "call": _steps, "score": s, "why": why, "seated": seated_in(sp), "through": through(sp, tb),
            "depth_err_mm": round(depth_error(sp, tb) * 1000, 2), "tube_bottom": [round(v, 4) for v in bottom(tb)],
            "tube_tilt": round(tilt_deg(tb), 2), "spinner": [round(v, 4) for v in shoulder(sp)],
            "pads_mm": {k: round(v * 1000, 2) for k, v in sp.q.items()},
            "contacts": {"tube": tb.contacts, "spinner": sp.contacts},
            "hands": {a: [round(float(v), 3) for v in (*h["tcp"], h["open"])] for a, h in tb.hands.items()},
            "ever": dict(_ever)}) + "\n")


def score(objects):
    """0 if at any time the tube was dropped or the spinner fell, a gripper held the tube in its lower 80 mm (below the
    red band) or another robot link touched the tube, the spinner was lifted off a seat without a gripper on its body
    (carried by the tube), or a neighbouring spinner was moved. Otherwise, at the end, with the tube through the spinner
    seated in the gauge or in the carousel's open position and both tool points 4 cm clear: 0.25; 0.5 with the column's
    centre within 3 mm of the line's depth; 0.75 within 1 mm; 1.0 within 1 mm with the spinner in the carousel."""
    global _steps
    _steps += 1
    _watch(objects)
    s, why = evaluate(objects)
    _trace(objects, s, why)
    return s


def success(objects):
    sp, tb = objects["spinner"], objects["tube"]
    still = np.linalg.norm(sp.lin_vel) < 0.005 and np.linalg.norm(tb.lin_vel) < 0.005
    return bool(evaluate(objects)[0] == 1.0 and still)


def distance(objects):
    return float(abs(depth_error(objects["spinner"], objects["tube"])))


# -- scripted demo ------------------------------------------------------------------------------------------------------
# A waypoint's ``yaw`` may be a full tool rotation (nine numbers, row by row) for the side grips used here.
_down_grip = _demo.grip_rotation


def _grip_rotation(yaw=0.0):
    return np.array(yaw, float).reshape(3, 3) if np.ndim(yaw) else _down_grip(yaw)


_demo.grip_rotation = _grip_rotation


def _tool(approach, closing) -> tuple:
    z = np.array(approach, float) / np.linalg.norm(approach)
    x = np.array(closing, float)
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    return tuple(np.c_[x, np.cross(z, x), z].reshape(-1))


def _side(base_y, xy, tilt_deg_):
    """A side grip from the arm's base towards ``xy``, tilted down by ``tilt_deg_``, the fingers closing level."""
    d = np.array([xy[0], xy[1] - base_y, 0.0])
    d /= np.linalg.norm(d)
    t = math.radians(tilt_deg_)
    return np.array([d[0] * math.cos(t), d[1] * math.cos(t), -math.sin(t)]), _tool(
        [d[0] * math.cos(t), d[1] * math.cos(t), -math.sin(t)], [d[1], -d[0], 0.0])


def _wp(t, p, finger, label, rot) -> Waypoint:
    return Waypoint(t, (float(p[0]), float(p[1]), float(p[2]) + TOP), finger, label, rot)


GRIP_AT = _env("GRIP_AT", 0.155)  # the tube is held this far above its bottom (in the sleeve)
PUSH_OFFSET = _env("PUSH_OFFSET", 0.0218)  # extra command below the target, for the arm's give while pushing (calibrated)
PUSH_DX, PUSH_DY = _env("PUSH_DX", -0.004), _env("PUSH_DY", -0.003)  # lateral aim correction for the arm's drift while pushing
TUBE_CLOSE = _env("TUBE_CLOSE", -0.0005)  # finger target closed on the 5 mm tube (pads just past touching)
SPIN_CLOSE = _env("SPIN_CLOSE", -0.005)  # ... on the turbine, 23 mm across its flats
SLOT_FIX = np.array([_env("SLOT_DX", 0.0), _env("SLOT_DY", 0.0), 0.0])  # aim correction at the changer (calibrated)
STAGE = os.environ.get("STAGE", "full")  # "insert": only the tube into the gauge (physics proof)
TOUCH = os.environ.get("TOUCH", "")  # bypass demos for the sim checks: "low" (hold below the band), "floor", "tube"


def demo():
    """Right hand: pick the tube from the rack by its sleeve (a side grip tilted 20 degrees down), lift it over the
    spinner, lower it into the bore and push it down until the column's centre is on the line, let go and back off.
    Left hand: grip the spinner's turbine from the side, lift it straight up until the tube is clear of the gauge, carry it
    round to the changer rack's open position between its neighbours, lower it onto the seat, let go and back off."""
    o, c = FINGER_OPEN, FINGER_CLOSED
    half = -0.022  # fingers about 44 mm apart
    zr, R = _side(-0.305, RACK_XY, 20.0)
    zg, RG = _side(-0.305, GAUGE_XY, 20.0)
    grip = GRIP_AT if TOUCH != "low" else 0.060
    target_bottom = LINE - COL_MID if TOUCH != "floor" else FLOOR - 0.003
    at_rack = np.array([*RACK_XY, 0.0005 + grip])
    lifted = np.array([*RACK_XY, 0.125 + grip])
    over = np.array([*GAUGE_XY, 0.125 + grip])
    mouth = np.array([*GAUGE_XY, SEAT + TURB_H + 0.004 + grip])
    pushed = np.array([GAUGE_XY[0] + PUSH_DX, GAUGE_XY[1] + PUSH_DY, target_bottom + grip - PUSH_OFFSET])
    right = [
        _wp(2.5, at_rack - 0.06 * zr, half, "behind the tube's sleeve", R),
        _wp(1.5, at_rack, half, "fingers either side of the sleeve", R),
        _wp(0.8, at_rack, TUBE_CLOSE, "grip the tube", R),
        _wp(2.0, lifted, TUBE_CLOSE, "lift it out of the rack", R),
        _wp(2.5, over, TUBE_CLOSE, "over the spinner", RG),
        _wp(1.5, mouth, TUBE_CLOSE, "down to the spinner's mouth", RG),
        _wp(2.5, mouth - np.array([0, 0, 0.015]), TUBE_CLOSE, "into the bore", RG),
        _wp(5.0, pushed, TUBE_CLOSE, "push it down to the line", RG),
        _wp(1.0, pushed, TUBE_CLOSE, "hold", RG),
        _wp(0.8, pushed, half, "let go", RG),
        _wp(1.5, pushed - 0.07 * zg, half, "back off", RG),
        _wp(2.0, (0.20, -0.25, 0.25), o, "away", RG),
    ]
    if STAGE == "insert":
        return {"right": right}
    t_left = sum(w.duration for w in right) - 3.0
    zl, RL = _side(0.305, GAUGE_XY, 20.0)
    zs, RS = _side(0.305, SLOT_XY, 20.0)
    turb = np.array([*GAUGE_XY, SEAT + TURB_H / 2 + 0.002])
    lift = SEAT + 0.105 - (LINE - COL_MID)  # the tube's bottom 10 mm above the gauge's top
    up = turb + np.array([0, 0, lift - SEAT + TURB_H / 2 - TURB_H / 2])
    slot_t = np.array([*SLOT_XY, SEAT + TURB_H / 2 + 0.002])
    slot_up = slot_t + (up - turb)
    left = [_wp(t_left - 2.5, (0.15, 0.16, 0.22), o, "wait", RL),
            _wp(2.5, (0.15, 0.0, 0.17), o, "round in front of the changer rack", RL)]
    if TOUCH == "tube":  # the bypass: lift the spinner by the tube (a grip on the tube's sleeve)
        tg = np.array([*GAUGE_XY, target_bottom + GRIP_AT])
        left += [_wp(2.0, tg - 0.06 * zl, half, "behind the tube's sleeve", RL),
                 _wp(1.2, tg, half, "fingers either side", RL), _wp(0.8, tg, TUBE_CLOSE, "grip the tube", RL),
                 _wp(2.0, tg + (up - turb), TUBE_CLOSE, "lift by the tube", RL)]
        return {"right": right, "left": left}
    left += [
        _wp(2.0, turb - 0.06 * zl, half, "beside the spinner", RL),
        _wp(1.5, turb, half, "fingers either side of the turbine", RL),
        _wp(0.8, turb, SPIN_CLOSE, "grip the turbine", RL),
        _wp(2.5, up, SPIN_CLOSE, "lift it straight up, the tube out of the gauge", RL),
        _wp(2.5, np.array([0.19, 0.02, up[2]]), SPIN_CLOSE, "back towards the robot, clear of the rack", RL),
        _wp(3.0, slot_up - 0.08 * np.array([_d[0], _d[1], 0.0]), SPIN_CLOSE, "across to just short of the rack", RS),
        _wp(3.0, slot_up + SLOT_FIX, SPIN_CLOSE, "in over the open position, between the neighbours' tubes", RS),
        _wp(1.5, slot_up + SLOT_FIX, SPIN_CLOSE, "let it settle", RS),
        _wp(4.0, slot_t + SLOT_FIX + np.array([0, 0, 0.0005]), SPIN_CLOSE, "lower it between the neighbours onto the seat", RS),
        _wp(0.8, slot_t + SLOT_FIX + np.array([0, 0, 0.0005]), half, "let go", RS),
        _wp(1.5, slot_t + SLOT_FIX - 0.06 * np.array([zs[0], zs[1], 0.0]), half, "back off level", RS),
        _wp(2.0, (0.18, 0.20, 0.22), o, "away", RS),
    ]
    T = max(sum(w.duration for w in left), sum(w.duration for w in right)) + 1.5
    for arm in (left, right):
        arm.append(Waypoint(T - sum(w.duration for w in arm), arm[-1].pos, arm[-1].finger, "wait", arm[-1].yaw))
    return {"right": right, "left": left}
