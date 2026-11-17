"""On the station table: four kiwis that look alike, lying in two small molded-pulp trays, two on the robot's left and
two on its right. Two of them are ripe and give when squeezed gently; the other two are still hard. Which two are
ripe changes from trial to trial (TASK_SEED 0-5). In the middle of the table stand a white fruit bowl (front) and a
brown paper bag with its top rolled down (near the robot). Ripe kiwis go in the bowl, to eat now; hard ones in the
bag, to ripen. A ripe kiwi squeezed hard (more than about 15 N, a 6 mm squeeze) is crushed and counts for nothing.

Run it:  TASK_SEED=0 inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "$(cat instruction.txt)" --scorer task_score -E object_jitter=0 -E control_hz=20 ...
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
# each finger (2 mm between two pads at 5 N); a hard one 0.05 mm
RIPE_C, HARD_C = 30.0, 150.0  # contact damping (N s/m)
KIWI = (0.47, 0.37, 0.21)  # fuzzy greenish-brown skin
CRUSH_N = 15.0  # a ripe kiwi pressed with more than this by any part of the robot is crushed (3 mm sink per contact)

FLOOR = 0.003  # the trays' floor thickness
KZ = TOP + FLOOR + SEMI[2]  # a kiwi's centre height in its pocket
TRAY_Y = 0.15  # the trays' centre lines, either side of the midline
POCKET_X = (0.22, 0.32)  # the near and far pockets' centres
POCKETS = {"kiwi_l_near": (POCKET_X[0], TRAY_Y), "kiwi_l_far": (POCKET_X[1], TRAY_Y),
           "kiwi_r_near": (POCKET_X[0], -TRAY_Y), "kiwi_r_far": (POCKET_X[1], -TRAY_Y)}
NAMES = tuple(POCKETS)
VARIANTS = tuple(itertools.combinations(NAMES, 2))  # the six ways two of the four can be ripe, TASK_SEED 0..5
SEED = int(os.environ.get("TASK_SEED", "0"))
if not 0 <= SEED < len(VARIANTS):
    raise ValueError(f"TASK_SEED must be 0..{len(VARIANTS) - 1}, got {SEED}")
RIPE = frozenset(VARIANTS[SEED])


def kiwis(ripe=RIPE):
    return [fruit(n, (x, y, KZ), SEMI, MASS, RIPE_K if n in ripe else HARD_K, RIPE_C if n in ripe else HARD_C, KIWI)
            for n, (x, y) in POCKETS.items()]


# -- the trays: molded pulp, two pockets each, fixed to the table ----------------------------------------------------

PULP = (0.80, 0.78, 0.72)
POCKET_IN = (0.076, 0.060)  # a pocket's inside, x by y
WALL_T, WALL_H = 0.004, 0.010  # pocket walls: thickness, height above the floor


def tray(side: str, y: float) -> list[Box]:
    x0, x1 = POCKET_X[0] - POCKET_IN[0] / 2 - WALL_T, POCKET_X[1] + POCKET_IN[0] / 2 + WALL_T
    wy = POCKET_IN[1] / 2 + WALL_T / 2
    zc = TOP + FLOOR + WALL_H / 2
    mid = (POCKET_X[0] + POCKET_X[1]) / 2
    return [
        Box(f"tray_{side}_floor", ((x0 + x1) / 2, y, TOP + FLOOR / 2), PULP, size=(x1 - x0, POCKET_IN[1] + 2 * WALL_T, FLOOR)),
        Box(f"tray_{side}_wall_l", ((x0 + x1) / 2, y + wy, zc), PULP, size=(x1 - x0, WALL_T, WALL_H)),
        Box(f"tray_{side}_wall_r", ((x0 + x1) / 2, y - wy, zc), PULP, size=(x1 - x0, WALL_T, WALL_H)),
        Box(f"tray_{side}_wall_near", (x0 + WALL_T / 2, y, zc), PULP, size=(WALL_T, POCKET_IN[1], WALL_H)),
        Box(f"tray_{side}_wall_far", (x1 - WALL_T / 2, y, zc), PULP, size=(WALL_T, POCKET_IN[1], WALL_H)),
        Box(f"tray_{side}_wall_mid", (mid, y, zc), PULP, size=(POCKET_X[1] - POCKET_X[0] - POCKET_IN[0], POCKET_IN[1], WALL_H)),
    ]


# -- the bowl (ripe kiwis) and the paper bag (hard ones) -------------------------------------------------------------

BOWL_XY = (0.36, 0.0)  # a white bowl 19 cm across at the rim, 9 cm across its foot, 6.5 cm deep
BOWL_RIM_R, BOWL_BASE_R, BOWL_DEPTH = 0.095, 0.045, 0.065
BOWL_RIM = TOP + BOWL_DEPTH
BOWL_IN_R = 0.080  # a kiwi's centre within this of the bowl's axis is in the bowl
BAG_X, BAG_Y = (0.125, 0.215), (-0.075, 0.075)  # the bag's inside (x, y)
BAG_H, PAPER = 0.10, 0.003  # rolled down to 10 cm tall; paper (as modelled) 3 mm
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
    *tray("l", TRAY_Y), *tray("r", -TRAY_Y),
    bowl("bowl", (*BOWL_XY, TOP), BOWL_BASE_R, BOWL_RIM_R, BOWL_DEPTH, (0.93, 0.93, 0.91)),
    *bag(),
    *kiwis(),
]

# -- scoring ---------------------------------------------------------------------------------------------------------

STILL = 0.02  # m/s
CONTACT_N = 0.05  # N: any robot link pressing on a kiwi harder than this is touching it
_crushed: set = set()  # ripe kiwis crushed at some step of the trial (one trial per process)
_TRACE = os.environ.get("TASK_TRACE")
_calls = 0


def reset_trial() -> None:
    """Forget crushed kiwis (one trial per process in the simulation; the tests call this)."""
    _crushed.clear()


def in_bowl(s) -> bool:
    d = math.hypot(s.pos[0] - BOWL_XY[0], s.pos[1] - BOWL_XY[1])
    return d <= BOWL_IN_R and TOP < s.pos[2] < BOWL_RIM


def in_bag(s) -> bool:
    return BAG_X[0] < s.pos[0] < BAG_X[1] and BAG_Y[0] < s.pos[1] < BAG_Y[1] and TOP < s.pos[2] < TOP + BAG_H


def touched(s) -> bool:
    """A robot part is on the kiwi now. If PhysX can't report contacts, a gripper's tool point within 6 cm counts."""
    if s.force is None:
        return any(np.linalg.norm(np.asarray(h["tcp"]) - s.pos) < 0.06 for h in s.hands.values())
    return any(f > CONTACT_N for f in s.force.values())


def settled(s) -> bool:
    return float(np.linalg.norm(s.lin_vel)) < STILL and not touched(s)


def update_crush(objects) -> None:
    for n in NAMES:
        s = objects[n]
        if n in RIPE and s.force is not None and s.robot_force > CRUSH_N:
            _crushed.add(n)


def verdict(objects, n) -> str:
    """Where kiwi ``n`` ended up: "right" (its own container, released, still, and not crushed if ripe), "wrong" (the
    other container, released and still), or "none" (anywhere else, still held or moving, or a crushed ripe kiwi in
    the bowl)."""
    s = objects[n]
    if not settled(s):
        return "none"
    bowl, bag_ = in_bowl(s), in_bag(s)
    if (bowl and n not in RIPE) or (bag_ and n in RIPE):
        return "wrong"
    if bowl and n in RIPE:
        return "none" if n in _crushed else "right"
    if bag_ and n not in RIPE:
        return "right"
    return "none"


def score(objects):
    """+0.25 for each kiwi released and still in its own container (ripe: the bowl, never crushed; hard: the bag);
    -0.25 for each kiwi released and still in the other one; 0 for a kiwi anywhere else (left in its tray, on the table,
    on the floor, held, moving) and for a crushed ripe kiwi in the bowl. The total is never below 0."""
    update_crush(objects)
    v = {n: verdict(objects, n) for n in NAMES}
    total = 0.25 * sum(1 for x in v.values() if x == "right") - 0.25 * sum(1 for x in v.values() if x == "wrong")
    _trace(objects, v, total)
    return max(0.0, total)


def success(objects):
    return score(objects) >= 1.0 - 1e-9


def distance(objects):
    """Kiwis not yet right, as a number (0 when all four are)."""
    return float(sum(1 for n in NAMES if verdict(objects, n) != "right"))


def _trace(objects, v, total) -> None:
    global _calls
    _calls += 1
    if not _TRACE or _calls % 6:
        return
    row = {"call": _calls, "seed": SEED, "score": round(max(0.0, total), 3), "crushed": sorted(_crushed)}
    for n in NAMES:
        s = objects[n]
        row[n] = {"ripe": n in RIPE, "xyz_mm": [round(float(c) * 1000) for c in s.pos], "v": v[n],
                  "f": s.force if s.force is None else {k.replace("_robot", ""): f for k, f in s.force.items()}}
    any_s = objects[NAMES[0]]
    row["hands"] = {a.replace("_robot", ""): {"tcp_mm": [round(float(c) * 1000) for c in h["tcp"]],
                                             "gap_mm": round(h["gap"] * 1000, 1)} for a, h in any_s.hands.items()}
    with open(_TRACE, "a") as f:
        f.write(json.dumps(row) + "\n")


# -- scripted demo ---------------------------------------------------------------------------------------------------
# Each arm checks the two kiwis on its own side: it closes on each one gently (each pad driven 2 mm past first touch:
# about 5 N on a ripe kiwi, which gives 2 mm, and 9.5 N on a hard one, which barely gives), holds the squeeze for a
# second, then lifts it with the same gentle grip and lets it go in the bowl or the bag. The plan reads which kiwis are
# ripe from the task (TASK_SEED), as a scripted demo may; the squeeze is still made on every kiwi.

FINGER_WIDE = -0.04695  # fully open
PAD_INSET = 0.0015  # measured in the simulation: the finger pads' colliders meet a kiwi 1.5 mm before the finger
# joints' nominal gap says they would (a hard kiwi stops them at a nominal gap of 49 mm, not 52)
SQUEEZE = -(SEMI[1] - PAD_INSET - 0.002)  # each pad driven 2 mm past first touch
LET_GO = -0.040
HOVER = TOP + 0.10
CARRY = TOP + 0.145
GRIP_Z = KZ + 0.002  # the tool point just above the kiwi's centre: the pad tips stay above the pocket walls
BOWL_DROP = (0.32, 0.03)  # where each arm lets a ripe kiwi go (y mirrored for the right arm)
BOWL_DROP_Z = BOWL_RIM + 0.012 + SEMI[2]
BAG_DROP = (0.17, 0.03)
BAG_DROP_Z = TOP + BAG_H + 0.012 + SEMI[2]
HARD_GRIP = os.environ.get("HARD_GRIP", "0") == "1"  # HARD_GRIP=1, the crush check in the simulation: the same demo
# with the fingers driven fully shut (the gripper's full 50 N) instead of the gentle squeeze; the ripe kiwis are crushed
# and score nothing, so it ends at 0.5 (the two hard kiwis in the bag)
STAGGER = float(os.environ.get("STAGGER", 6.0))  # the right arm starts this much later, so the arms never meet mid-table


def check_and_sort(name: str, ripe: bool, side: float) -> list[Waypoint]:
    x, y = POCKETS[name]
    dx, dy = BOWL_DROP if ripe else BAG_DROP
    dz = BOWL_DROP_Z if ripe else BAG_DROP_Z
    dy *= side
    where = "bowl" if ripe else "bag"
    grip = 0.0 if HARD_GRIP else SQUEEZE
    return [
        Waypoint(1.6, (x, y, HOVER), FINGER_WIDE, f"over {name}"),
        Waypoint(1.4, (x, y, GRIP_Z), FINGER_WIDE, "down round it"),
        Waypoint(1.0, (x, y, GRIP_Z), grip, "squeeze gently"),
        Waypoint(1.0, (x, y, GRIP_Z), grip, "hold the squeeze: feel it"),
        Waypoint(1.2, (x, y, CARRY), grip, "lift"),
        Waypoint(1.8, (dx, dy, CARRY), grip, f"carry to the {where}"),
        Waypoint(1.0, (dx, dy, dz), grip, "lower"),
        Waypoint(0.6, (dx, dy, dz), LET_GO, "let go"),
        Waypoint(1.0, (dx, dy + side * 0.06, CARRY), LET_GO, "up and back to its side"),
    ]


def demo():
    plan = {}
    for arm, side, names in (("left", 1.0, ("kiwi_l_far", "kiwi_l_near")), ("right", -1.0, ("kiwi_r_far", "kiwi_r_near"))):
        w = [] if side > 0 else [Waypoint(STAGGER, (0.22, -0.20, TOP + 0.16), FINGER_WIDE, "wait for the left arm")]
        for n in names:
            w += check_and_sort(n, n in RIPE, side)
        w.append(Waypoint(1.5, (0.20, side * 0.24, TOP + 0.16), FINGER_WIDE, "clear of the middle"))
        plan[arm] = w
    return plan
