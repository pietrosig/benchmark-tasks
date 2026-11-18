"""On the station table: eight kiwis that look alike, lying in two foam-board pocket trays, four on the robot's left
and four on its right. Four of them are ripe and give when squeezed gently; the other four are still hard. Which four
are ripe changes from trial to trial (TASK_SEED 0-69: every way of choosing four of the eight pockets). In the middle
of the table stand a white fruit bowl (far) and a brown paper bag with its top rolled down (near the robot). Ripe kiwis
go in the bowl, to eat now; hard ones in the bag, to ripen.

A kiwi counts only if it was squeezed in its pocket first (both pads of one gripper pressing it for 0.5 s before it
left the pocket), so a robot that drops kiwis in without feeling them earns nothing. A ripe kiwi pressed with more
than 15 N (a 6 mm squeeze) is crushed and counts for nothing. The score rises only once the right placements outnumber
the wrong ones by more than four, so guessing scores almost nothing (README: the exact chances).

Run it (README: the full commands and the instruction):  TASK_SEED=28 inspect-robots run --embodiment yam_arena \\
           -E task=task.py --instruction "$INSTR" --scorer task_score -E object_jitter=0 -E control_hz=20 ...
"""

from __future__ import annotations

import itertools
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from fruitkit import bowl, fruit  # noqa: E402  (importing fruitkit hooks yam_arena so the kiwis are soft)
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

# -- the kiwis -------------------------------------------------------------------------------------------------------

SEMI = (0.034, 0.026, 0.026)  # semi-axes (m): 68 mm long, 52 mm across, lying with the long axis forward (x)
MASS = 0.10  # kg
RIPE_K, HARD_K = 5000.0, 100000.0  # compliant-contact stiffness per contact (N/m): a ripe kiwi gives 1 mm per 5 N at
# each contact (2 mm between two pads at 5 N); a hard one 0.05 mm
RIPE_C, HARD_C = 30.0, 150.0  # contact damping (N s/m)
KIWI = (0.47, 0.37, 0.21)  # fuzzy greenish-brown skin
CRUSH_N = 15.0  # a ripe kiwi pressed with more than this by any part of the robot is crushed (3 mm sink per contact)

FLOOR = 0.003  # the trays' floor thickness
KZ = TOP + FLOOR + SEMI[2]  # a kiwi's centre height in its pocket
ROW_X = (0.23, 0.32)  # the near and far rows of pockets
COL_Y = (0.14, 0.22)  # the inner and outer columns, either side of the midline
# pocket codes: L1 left near inner, L2 left near outer, L3 left far inner, L4 left far outer; R1-R4 the same on the right
POCKETS = {}
for _side, _sy in (("L", 1.0), ("R", -1.0)):
    for _i, (_x, _y) in enumerate(((ROW_X[0], COL_Y[0]), (ROW_X[0], COL_Y[1]), (ROW_X[1], COL_Y[0]), (ROW_X[1], COL_Y[1]))):
        POCKETS[f"kiwi_{_side}{_i + 1}"] = (_x, _sy * _y)
NAMES = tuple(POCKETS)
N_RIPE = 4
VARIANTS = tuple(itertools.combinations(NAMES, N_RIPE))  # the 70 ways four of the eight can be ripe, TASK_SEED 0..69
SEED = int(os.environ.get("TASK_SEED", "0"))
if not 0 <= SEED < len(VARIANTS):
    raise ValueError(f"TASK_SEED must be 0..{len(VARIANTS) - 1}, got {SEED}")
RIPE = frozenset(VARIANTS[SEED])


def kiwis(ripe=RIPE):
    return [fruit(n, (x, y, KZ), SEMI, MASS, RIPE_K if n in ripe else HARD_K, RIPE_C if n in ripe else HARD_C, KIWI)
            for n, (x, y) in POCKETS.items()]


# -- the trays: white foam board, four pockets each (2 x 2), taped to the table --------------------------------------

BOARD = (0.90, 0.90, 0.88)
POCKET_IN = (0.075, 0.060)  # a pocket's inside, x by y
WALL_T, WALL_H = 0.005, 0.010  # outer walls: thickness, height above the floor


def tray(side: str, sy: float) -> list[Box]:
    x0, x1 = ROW_X[0] - POCKET_IN[0] / 2 - WALL_T, ROW_X[1] + POCKET_IN[0] / 2 + WALL_T
    y_in, y_out = COL_Y[0] - POCKET_IN[1] / 2 - WALL_T, COL_Y[1] + POCKET_IN[1] / 2 + WALL_T
    cx, cy = (x0 + x1) / 2, sy * (y_in + y_out) / 2
    lx, ly = x1 - x0, y_out - y_in
    zc = TOP + FLOOR + WALL_H / 2
    gx0, gx1 = ROW_X[0] + POCKET_IN[0] / 2, ROW_X[1] - POCKET_IN[0] / 2  # the divider between the rows
    gy0, gy1 = COL_Y[0] + POCKET_IN[1] / 2, COL_Y[1] - POCKET_IN[1] / 2  # the divider between the columns
    return [
        Box(f"tray_{side}_floor", (cx, cy, TOP + FLOOR / 2), BOARD, size=(lx, ly, FLOOR)),
        Box(f"tray_{side}_wall_in", (cx, sy * (y_in + WALL_T / 2), zc), BOARD, size=(lx, WALL_T, WALL_H)),
        Box(f"tray_{side}_wall_out", (cx, sy * (y_out - WALL_T / 2), zc), BOARD, size=(lx, WALL_T, WALL_H)),
        Box(f"tray_{side}_wall_near", (x0 + WALL_T / 2, cy, zc), BOARD, size=(WALL_T, ly - 2 * WALL_T, WALL_H)),
        Box(f"tray_{side}_wall_far", (x1 - WALL_T / 2, cy, zc), BOARD, size=(WALL_T, ly - 2 * WALL_T, WALL_H)),
        Box(f"tray_{side}_div_rows", ((gx0 + gx1) / 2, cy, zc), BOARD, size=(gx1 - gx0, ly - 2 * WALL_T, WALL_H)),
        Box(f"tray_{side}_div_cols", (cx, sy * (gy0 + gy1) / 2, zc), BOARD, size=(lx - 2 * WALL_T, gy1 - gy0, WALL_H)),
    ]


# -- the bowl (ripe kiwis) and the paper bag (hard ones) -------------------------------------------------------------

BOWL_XY = (0.43, 0.0)  # a white serving bowl 23 cm across at the rim, 11 cm across its foot, 8 cm deep
BOWL_RIM_R, BOWL_BASE_R, BOWL_DEPTH = 0.115, 0.055, 0.080
BOWL_RIM = TOP + BOWL_DEPTH
BOWL_IN_R = 0.095  # a kiwi's centre within this of the bowl's axis (and below the rim) is in the bowl
BAG_X, BAG_Y = (0.045, 0.159), (-0.0895, 0.0895)  # the bag's inside (x, y): a #12 bag's 11.4 x 17.9 cm base, its far
# side 6 cm short of the near pockets' centres so the gripper's body clears its rolled top over the near inner kiwis
BAG_H, PAPER = 0.12, 0.003  # rolled down to 12 cm tall; paper (as modelled) 3 mm
KRAFT = (0.66, 0.50, 0.33)


def bag() -> list[Box]:
    (x0, x1), (y0, y1) = BAG_X, BAG_Y
    cx, cy, w, d = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    zc = TOP + BAG_H / 2
    return [
        Box("bag_bottom", (cx, cy, TOP + PAPER / 2), KRAFT, size=(w, d, PAPER)),
        Box("bag_front", (x1 + PAPER / 2, cy, zc), KRAFT, size=(PAPER, d + 2 * PAPER, BAG_H)),
        Box("bag_back", (x0 - PAPER / 2, cy, zc), KRAFT, size=(PAPER, d + 2 * PAPER, BAG_H)),
        Box("bag_left", (cx, y1 + PAPER / 2, zc), KRAFT, size=(w, PAPER, BAG_H)),
        Box("bag_right", (cx, y0 - PAPER / 2, zc), KRAFT, size=(w, PAPER, BAG_H)),
        # the rolled-down top: a thicker band of folded paper round the opening
        Box("bag_roll_front", (x1 + 0.004, cy, TOP + BAG_H - 0.008), KRAFT, size=(0.008, d + 0.016, 0.016)),
        Box("bag_roll_back", (x0 - 0.004, cy, TOP + BAG_H - 0.008), KRAFT, size=(0.008, d + 0.016, 0.016)),
        Box("bag_roll_left", (cx, y1 + 0.004, TOP + BAG_H - 0.008), KRAFT, size=(w, 0.008, 0.016)),
        Box("bag_roll_right", (cx, y0 - 0.004, TOP + BAG_H - 0.008), KRAFT, size=(w, 0.008, 0.016)),
    ]


OBJECTS = [
    *tray("l", 1.0), *tray("r", -1.0),
    bowl("bowl", (*BOWL_XY, TOP), BOWL_BASE_R, BOWL_RIM_R, BOWL_DEPTH, (0.93, 0.93, 0.91)),
    *bag(),
    *kiwis(),
]

# -- scoring ---------------------------------------------------------------------------------------------------------
# Each kiwi, at the end: "right" (at rest in its own container, let go, squeezed in its pocket first, and not crushed if
# ripe), "wrong" (at rest in the other container, let go: squeezed or not), or "none" (anywhere else; held or moving;
# in its own container but never squeezed in its pocket; a crushed ripe kiwi in the bowl).
# net = rights - wrongs; score = (net - 4) / 4, clipped to 0..1: 0.25 at net 5, 0.5 at 6, 0.75 at 7, 1.0 at 8.

STILL = 0.02  # m/s
CONTACT_N = 0.05  # N: any robot link pressing on a kiwi harder than this is touching it
SQUEEZE_N = 1.0  # N: each pad presses at least this ...
SQUEEZE_S = 0.5  # s: ... both pads of one gripper at once, for this long in all, while the kiwi is in its pocket
POCKET_R, POCKET_DZ = 0.025, 0.010  # in its pocket: centre within 2.5 cm (horizontally) of where it started, < 1 cm up
NET_FREE = 4  # the score starts once the right placements outnumber the wrong ones by more than this
FALLBACK_DT = 0.025  # if the simulation's clock is unknown: score() runs twice per 20 Hz control step

_crushed: set = set()  # ripe kiwis crushed at some step of the trial (one trial per process)
_squeezed: set = set()  # kiwis that have had their squeeze in their pocket
_squeeze_s: dict = {}  # kiwi -> seconds of two-pad squeeze in its pocket so far
_start: dict = {}  # kiwi -> where it was first seen (its place in the pocket)
_last_t: list = [None]
_TRACE = os.environ.get("TASK_TRACE")
_calls = 0


def reset_trial() -> None:
    """Forget the trial so far (one trial per process in the simulation; the tests call this)."""
    _crushed.clear()
    _squeezed.clear()
    _squeeze_s.clear()
    _start.clear()
    _last_t[0] = None


def in_bowl(s) -> bool:
    d = math.hypot(s.pos[0] - BOWL_XY[0], s.pos[1] - BOWL_XY[1])
    return d <= BOWL_IN_R and TOP < s.pos[2] < BOWL_RIM


def in_bag(s) -> bool:
    return BAG_X[0] < s.pos[0] < BAG_X[1] and BAG_Y[0] < s.pos[1] < BAG_Y[1] and TOP < s.pos[2] < TOP + BAG_H


def in_pocket(n, s) -> bool:
    p0 = _start.get(n)
    if p0 is None:
        return False
    return math.hypot(s.pos[0] - p0[0], s.pos[1] - p0[1]) < POCKET_R and s.pos[2] < p0[2] + POCKET_DZ


def two_pad_squeeze(s) -> bool:
    """Both finger pads of one gripper press the kiwi with at least SQUEEZE_N each, right now."""
    if not s.force:
        return False
    for arm in ("left_robot", "right_robot"):
        if all(s.force.get(f"{arm}/{pad}", 0.0) >= SQUEEZE_N for pad in ("tip_left", "tip_right")):
            return True
    return False


def touched(s) -> bool:
    """A robot part is on the kiwi now. If PhysX can't report contacts, a gripper's tool point within 6 cm counts."""
    if s.force is None:
        return any(np.linalg.norm(np.asarray(h["tcp"]) - s.pos) < 0.06 for h in s.hands.values())
    return any(f > CONTACT_N for f in s.force.values())


def settled(s) -> bool:
    return float(np.linalg.norm(s.lin_vel)) < STILL and not touched(s)


def update(objects) -> None:
    """Advance the trial's memory by one call: where each kiwi started, its squeeze time in its pocket, crushing."""
    t = getattr(objects[NAMES[0]], "t", None)
    if t is None:
        dt = FALLBACK_DT
    else:
        dt = 0.0 if _last_t[0] is None else max(0.0, float(t) - _last_t[0])
        _last_t[0] = float(t)
    for n in NAMES:
        s = objects[n]
        if n not in _start:
            _start[n] = np.array(s.pos, dtype=float)
        if n in RIPE and s.force is not None and s.robot_force > CRUSH_N:
            _crushed.add(n)
        if n not in _squeezed and in_pocket(n, s) and two_pad_squeeze(s):
            _squeeze_s[n] = _squeeze_s.get(n, 0.0) + dt
            if _squeeze_s[n] >= SQUEEZE_S - 1e-9:
                _squeezed.add(n)


def verdict(objects, n) -> str:
    s = objects[n]
    if not settled(s):
        return "none"
    bowl_, bag_ = in_bowl(s), in_bag(s)
    if (bowl_ and n not in RIPE) or (bag_ and n in RIPE):
        return "wrong"
    if not (bowl_ or bag_) or n not in _squeezed:
        return "none"
    if bowl_ and n in _crushed:
        return "none"
    return "right"


def net_to_score(net: int) -> float:
    return min(1.0, max(0.0, (net - NET_FREE) / (len(NAMES) - NET_FREE)))


def score(objects):
    """(rights - wrongs - 4) / 4, clipped to 0..1 (see above)."""
    update(objects)
    v = {n: verdict(objects, n) for n in NAMES}
    net = sum(x == "right" for x in v.values()) - sum(x == "wrong" for x in v.values())
    total = net_to_score(net)
    _trace(objects, v, net, total)
    return total


def success(objects):
    return score(objects) >= 1.0 - 1e-9


def distance(objects):
    """Kiwis not yet right, as a number (0 when all eight are)."""
    return float(sum(1 for n in NAMES if verdict(objects, n) != "right"))


def _trace(objects, v, net, total) -> None:
    global _calls
    _calls += 1
    if not _TRACE or _calls % 6:
        return
    s0 = objects[NAMES[0]]
    row = {"call": _calls, "t": getattr(s0, "t", None), "seed": SEED, "score": round(total, 3), "net": net,
           "crushed": sorted(_crushed), "squeezed": sorted(_squeezed)}
    for n in NAMES:
        s = objects[n]
        row[n] = {"ripe": n in RIPE, "xyz_mm": [round(float(c) * 1000) for c in s.pos], "v": v[n],
                  "sq_s": round(_squeeze_s.get(n, 0.0), 2),
                  "f": s.force if s.force is None else {k.replace("_robot", ""): f for k, f in s.force.items()}}
    row["hands"] = {a.replace("_robot", ""): {"tcp_mm": [round(float(c) * 1000) for c in h["tcp"]],
                                             "gap_mm": round(h["gap"] * 1000, 1)} for a, h in s0.hands.items()}
    with open(_TRACE, "a") as f:
        f.write(json.dumps(row) + "\n")


# -- scripted demo ---------------------------------------------------------------------------------------------------
# Each arm checks the four kiwis on its own side, one at a time: it opens to 72 mm (clear of the neighbouring kiwis),
# goes down round the kiwi, closes gently (each pad driven 2 mm past first touch: about 6 N on a ripe kiwi, which gives
# 2 mm, and 9.5 N on a hard one, which barely gives), holds the squeeze for a second in the pocket, then lifts it with
# the same gentle grip and lets it go in the bowl or the bag. The plan reads which kiwis are ripe from the task
# (TASK_SEED), as a scripted demo may; the squeeze is made on every kiwi the same way.

FINGER_WIDE = -0.036  # pads 72 mm apart: 10 mm clear of the kiwi each side, clear of the next pocket's kiwi
FINGER_OPEN_FULL = -0.04695
PAD_INSET = 0.0015  # measured in the simulation: the finger pads' colliders meet a kiwi 1.5 mm before the finger
# joints' nominal gap says they would (a hard kiwi stops them at a nominal gap of 49 mm, not 52)
SQUEEZE = -(SEMI[1] - PAD_INSET - 0.002)  # each pad driven 2 mm past first touch
LET_GO = FINGER_WIDE
HOVER = TOP + 0.145  # above the bag's rolled top (12 cm) with the pad tips 1.2 cm clear: the arms pass it between kiwis
CARRY_BAG = TOP + 0.165  # carrying height to the bag (its rolled top stands 12 cm) ...
CARRY_BOWL = TOP + 0.120  # ... and to the bowl (rim 8 cm; the far reach of the arm is lower)
GRIP_Z = KZ + 0.002  # the tool point just above the kiwi's centre: the pad tips stay above the pocket walls
BOWL_DROP = ((0.360, 0.055), (0.370, 0.035))  # where each arm lets its ripe kiwis go (y mirrored for the right arm)
BOWL_DROP_Z = BOWL_RIM + 0.012 + SEMI[2]
BAG_DROP = ((0.102, 0.050), (0.102, 0.020))
BAG_DROP_Z = TOP + BAG_H + 0.012 + SEMI[2]
HARD_GRIP = os.environ.get("HARD_GRIP", "0") == "1"  # HARD_GRIP=1, the crush check in the simulation: the same demo
# with the fingers driven fully shut (the gripper's full 50 N) instead of the gentle squeeze; the ripe kiwis are crushed
STAGGER = float(os.environ.get("STAGGER", 5.5))  # the right arm starts this much later, so the arms never meet mid-table
ORDER = ("4", "3", "2", "1")  # far outer, far inner, near outer, near inner
WITHDRAW = (0.28, 0.30, TOP + 0.22)  # where each arm ends, out over its own side (y mirrored for the right arm)


def check_and_sort(name: str, ripe: bool, side: float, k: int) -> list[Waypoint]:
    x, y = POCKETS[name]
    dx, dy = (BOWL_DROP if ripe else BAG_DROP)[k % 2]
    dz = BOWL_DROP_Z if ripe else BAG_DROP_Z
    carry = CARRY_BOWL if ripe else CARRY_BAG
    dy *= side
    where = "bowl" if ripe else "bag"
    grip = 0.0 if HARD_GRIP else SQUEEZE
    return [
        Waypoint(1.5, (x, y, HOVER), FINGER_WIDE, f"over {name}"),
        Waypoint(1.2, (x, y, GRIP_Z), FINGER_WIDE, "down round it"),
        Waypoint(0.8, (x, y, GRIP_Z), grip, "squeeze gently"),
        Waypoint(1.0, (x, y, GRIP_Z), grip, "hold the squeeze in the pocket: feel it"),
        Waypoint(1.0, (x, y, carry), grip, "lift"),
        Waypoint(1.6, (dx, dy, carry), grip, f"carry to the {where}"),
        Waypoint(0.8, (dx, dy, dz), grip, "lower"),
        Waypoint(0.5, (dx, dy, dz), LET_GO, "let go"),
        Waypoint(0.9, (dx, dy + side * 0.08, carry), LET_GO, "up and back to its side"),
    ]


def demo():
    plan = {}
    for arm, side, code in (("left", 1.0, "L"), ("right", -1.0, "R")):
        w = [] if side > 0 else [Waypoint(STAGGER, (0.24, -0.23, TOP + 0.16), FINGER_WIDE, "wait for the left arm")]
        ripe_k = hard_k = 0
        for i in ORDER:
            n = f"kiwi_{code}{i}"
            ripe = n in RIPE
            w += check_and_sort(n, ripe, side, ripe_k if ripe else hard_k)
            if ripe:
                ripe_k += 1
            else:
                hard_k += 1
        # after its last kiwi the arm goes straight back out to its own side, high, instead of hovering over the
        # middle: by the time the last kiwi has settled for the 1 s success hold, both bowl and bag are in plain view
        w[-1] = Waypoint(0.9, (WITHDRAW[0], side * WITHDRAW[1], WITHDRAW[2]), FINGER_OPEN_FULL, "withdraw to its side")
        plan[arm] = w
    return plan
