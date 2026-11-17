"""A full-size e-bike frame (step-through, aluminium, with its mid-drive unit) is held on the bench as in a repair
stand, its down tube rising 25 degrees to the left. On top of the down tube are the two holders of a frame battery:
the lower (plug-side) holder just above the drive unit, whose pocket takes the contact block on the battery's bottom
end, and the upper (lock-side) holder 34 cm further up. The battery (338 x 93 x 81 mm, 3.0 kg) lies on its side on
the bench in front of the frame. It goes back on bottom end first: its contact block slides into the lower holder's
pocket (the battery can then be tilted no more than about 7 degrees off the tube), and the battery is swung down
about it into the upper holder and pressed until the latch clicks. A battery that falls scores 0.

The scripted demo (``--policy yam_scripted``) holds the battery with both hands across its depth and pre-deflects
its joint targets against the battery's weight (the simulated arms compensate their own weight, not a load).

Run it:  inspect-robots run --embodiment yam_arena --policy <policy> -E task=task.py \\
           --instruction "Put the battery back on the e-bike." \\
           --scorer task_score -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 --max-steps 2000
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
import ebkit  # noqa: E402  (importing the kit hooks yam_arena so its fixtures are simulated)
from ebkit import Slider, fixture  # noqa: E402
from yam_arena import ik  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"


def _env(name, default):
    return float(os.environ.get(name, default))


# -- geometry ------------------------------------------------------------------------------------------------------------
# Bench coordinates: x forward from the midpoint between the arm bases, y toward the left arm, z above the bench top.
# The frame's own coordinates ("tube frame"): X lateral (= bench x), Y up along the down tube, Z out of its top surface.
# Its origin is the battery's seat: the lower back corner of the battery's bottom end when the battery is home.
ALPHA = math.radians(25.0)  # the down tube rises this far above level, toward the left arm
SEAT = np.array([0.34, -0.15, 0.17])  # the seat point (bench coordinates)
CA, SA = math.cos(ALPHA), math.sin(ALPHA)
R_TUBE = np.array([[1.0, 0.0, 0.0], [0.0, CA, -SA], [0.0, SA, CA]])  # tube frame -> bench


def _qx(a):
    return (math.sin(a / 2), 0.0, 0.0, math.cos(a / 2))  # xyzw


# the battery: Bosch PowerPack Frame 540 (BBP3553), 338 x 93 x 81 mm, 3.0 kg
BL, BW, BD = 0.338, 0.093, 0.081  # length (along the tube), width (lateral), depth (off the tube)
BAT_KG = 3.0
NOTCH_Y = 0.318  # the latch notch in the battery's top end: from here to the end ...
NOTCH_Z = (0.005, 0.017)  # ... and this far off its back face
TONGUE = ((-0.025, 0.025), (-0.020, 0.0), (0.0025, 0.0225))  # the contact block on its bottom end: X, Y, Z ranges
# the lower holder's pocket
POCKET_END = -0.0215  # its end wall (the contact block's tip stops 1.5 mm short of it)
LIP_FRONT, LIP_LOW = -0.003, 0.0255  # its lip: front face and underside
POCKET_SIDE = 0.028  # its side walls (inner faces at +-28 mm; the block is 50 mm wide)
# the upper holder
LOCK_FACE = 0.342  # the lock housing's face, 4 mm past the battery's top end ...
LOCK_TOP = 0.020  # ... and its top, 20 mm off the seat (a battery tilted 5 degrees or more clears it)
PLUNGER_Y = 0.3075  # the latch's spring plunger under the battery's top end
PLUNGER_TRAVEL = 0.006
PLUNGER_K = 2500.0  # N/m, preloaded: 15 N with the plunger up, 30 N pressed flush
BOLT_TRAVEL = 0.0225

BLACK, CAP, LED, FRAME_C, MOTOR_C, HOLDER_C, GOLD, STAND_C, ALU = (
    (0.09, 0.09, 0.10), (0.24, 0.24, 0.26), (0.20, 0.75, 0.85), (0.16, 0.33, 0.30), (0.12, 0.12, 0.13),
    (0.20, 0.20, 0.22), (0.80, 0.66, 0.30), (0.75, 0.15, 0.12), (0.62, 0.63, 0.65))
VIS = {"collide": False}


def _box(lo, hi, color, **o):
    lo, hi = np.array(lo, float), np.array(hi, float)
    return (tuple((lo + hi) / 2), tuple(hi - lo), color, o) if o else (tuple((lo + hi) / 2), tuple(hi - lo), color)


def _to_tube(v):
    return R_TUBE.T @ np.asarray(v, float)


def _wcyl(p0, p1, r, color, collide=True):
    """A tube between two bench points (lateral offset kept), as a shape in the tube frame."""
    a, b = _to_tube(np.asarray(p0) - SEAT), _to_tube(np.asarray(p1) - SEAT)
    d = b - a
    ln = float(np.linalg.norm(d))
    rx = math.degrees(math.atan2(d[2], d[1]))
    o = {"shape": "cyl", "axis": "Y", "rx": rx}
    if not collide:
        o["collide"] = False
    return (tuple((a + b) / 2), (2 * r, ln, 2 * r), color, o)


def _wbox(center, size, color, collide=True):
    """A box level on the bench (size x, y, z), as a shape in the tube frame."""
    o = {"rx": -math.degrees(ALPHA)}
    if not collide:
        o["collide"] = False
    return (tuple(_to_tube(np.asarray(center) - SEAT)), tuple(size), color, o)


X0, Y0, Z0 = SEAT
BB = np.array([X0, Y0 - 0.16, Z0 - 0.10])  # the drive unit's crank axle
_frame = [
    # down tube: 64 mm, from the drive unit up to the head tube
    ((0.0, 0.25, -0.038), (0.064, 0.72, 0.064), FRAME_C, {"shape": "cyl", "axis": "Y"}),
    # drive unit (a Bosch mid-drive housing), level, the down tube running into its front
    _wbox((X0, Y0 - 0.12, Z0 - 0.09), (0.10, 0.20, 0.14), MOTOR_C),
    _wbox((X0, BB[1], BB[2]), (0.135, 0.03, 0.03), ALU, collide=False),  # crank axle stubs (cranks off)
    # head tube (70 degrees), seat tube (73 degrees), chain stays and seat stays toward the rear dropouts
    _wcyl((X0, Y0 + 0.555, Z0 + 0.18), (X0, Y0 + 0.50, Z0 + 0.33), 0.026, FRAME_C),
    _wcyl(BB + (0, -0.01, 0.05), BB + (0, -0.13, 0.47), 0.018, FRAME_C),
    _wcyl(BB + (0.045, -0.04, 0.0), BB + (0.065, -0.46, 0.03), 0.010, FRAME_C),
    _wcyl(BB + (-0.045, -0.04, 0.0), BB + (-0.065, -0.46, 0.03), 0.010, FRAME_C),
    _wcyl(BB + (0.03, -0.11, 0.40), BB + (0.065, -0.46, 0.03), 0.008, FRAME_C),
    _wcyl(BB + (-0.03, -0.11, 0.40), BB + (-0.065, -0.46, 0.03), 0.008, FRAME_C),
    # seat post and the repair stand's clamp on it, its arm going back to the stand's mast behind the bench (looks only)
    _wcyl(BB + (0, -0.125, 0.45), BB + (0, -0.16, 0.58), 0.0135, ALU, collide=False),
    _wbox(BB + (0, -0.15, 0.55), (0.06, 0.07, 0.07), STAND_C, collide=False),
    _wcyl(BB + (0.03, -0.15, 0.55), BB + (0.62, -0.15, 0.55), 0.016, STAND_C, collide=False),
    # lower (plug-side) holder: base plate, end wall, lip over the pocket, side walls; contacts inside (looks only)
    _box((-0.035, -0.045, -0.006), (0.035, 0.040, 0.0), HOLDER_C),
    _box((-0.035, -0.045, 0.0), (0.035, POCKET_END, 0.045), HOLDER_C),
    _box((-0.035, POCKET_END, LIP_LOW), (0.035, LIP_FRONT, 0.045), HOLDER_C),
    _box((POCKET_SIDE, POCKET_END, 0.0), (0.035, LIP_FRONT, LIP_LOW), HOLDER_C),
    _box((-0.035, POCKET_END, 0.0), (-POCKET_SIDE, LIP_FRONT, LIP_LOW), HOLDER_C),
    _box((-0.015, POCKET_END, 0.008), (0.015, POCKET_END + 0.004, 0.016), GOLD, collide=False),
    # upper (lock-side) holder: base plate and lock housing, with the key slot (looks only)
    _box((-0.03, 0.28, -0.006), (0.03, 0.38, 0.0), HOLDER_C),
    _box((-0.03, LOCK_FACE, 0.0), (0.03, 0.38, LOCK_TOP), HOLDER_C),
    _box((-0.0305, 0.356, 0.005), (-0.029, 0.368, 0.015), ALU, collide=False),
]
_DOT = (((0, 0, 0), (0.001, 0.001, 0.001), HOLDER_C, VIS),)
BIKE = fixture("bike", (X0, Y0, Z0 + TOP), _frame, [
    # the latch: a spring plunger under the battery's top end (pressed flush by a battery that is home) ...
    Slider("plunger", (0.0, PLUNGER_Y, PLUNGER_TRAVEL), "Z", (((0, 0, -0.006), (0.02, 0.015, 0.012), ALU),),
           low=-PLUNGER_TRAVEL, high=0.0, stiffness=PLUNGER_K, damping=20.0, mass=0.02, gravity=False,
           target=PLUNGER_TRAVEL),
    # ... and the bolt that then slides out of the lock housing into the notch in the battery's top end
    Slider("bolt", (0.0, 0.3505, 0.011), "Y", (((0, 0, 0), (0.016, 0.015, 0.008), ALU),), low=-BOLT_TRAVEL, high=0.0,
           stiffness=400.0, damping=10.0, mass=0.02, gravity=False, target=0.0, retarget=("@latched", 0.0, -BOLT_TRAVEL)),
], rot=_qx(ALPHA), friction=0.6)

# the battery, in its own frame (the same axes as the tube frame when it is home)
_bat = [
    _box((-BW / 2, 0.0, 0.0), (BW / 2, NOTCH_Y, BD), BLACK),
    _box((-BW / 2, NOTCH_Y, 0.0), (BW / 2, BL, NOTCH_Z[0]), CAP),
    _box((-BW / 2, NOTCH_Y, NOTCH_Z[1]), (BW / 2, BL, BD), CAP),
    _box((0.012, NOTCH_Y, NOTCH_Z[0]), (BW / 2, BL, NOTCH_Z[1]), CAP),
    _box((-BW / 2, NOTCH_Y, NOTCH_Z[0]), (-0.012, BL, NOTCH_Z[1]), CAP),
    _box(*zip(*TONGUE), HOLDER_C),
    _box((-BW / 2, 0.0, 0.0), (BW / 2, 0.012, BD), CAP, collide=False),  # bottom end cap
    _box((-0.012, 0.20, BD), (0.012, 0.26, BD + 0.001), LED, collide=False),  # charge LEDs
    _box((-0.0475, 0.03, 0.03), (-0.0465, 0.29, 0.05), CAP, collide=False),  # grip band on the flanks
    _box((0.0465, 0.03, 0.03), (0.0475, 0.29, 0.05), CAP, collide=False),
]
# at the start it lies on its right-hand flank on the bench, its back (the side that goes on the tube) toward the arms
START_POS = np.array([0.16, -BL / 2, BW / 2 + 0.001])
START_R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # battery frame -> bench: X down, Y left, Z forward
BATTERY = fixture("battery", tuple(START_POS + np.array([0.0, 0.0, TOP])), _bat, [], fixed=False, base_mass=BAT_KG,
                  friction=0.6, rot=(0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)))
OBJECTS = [BIKE, BATTERY]

# -- scoring -------------------------------------------------------------------------------------------------------------
_TRACE = os.environ.get("TASK_TRACE")
_calls = 0
_latched = [False]
_dropped = [False]
SEAT_AXIAL = 0.003  # bottom end home: within 3 mm of home along the tube (the contact block in the pocket) ...
SEAT_LATERAL = 0.004  # ... and laterally
SEAT_TILT = math.radians(8.0)  # ... tilted off the tube at most this far (the pocket allows about 7 degrees)
SWUNG_IN = 0.010  # top end within 10 mm of home: swung in
LATCH_TOP = 0.0015  # pressed home: the top end within 1.5 mm of home (the plunger flush to within 1 mm)
DROP_SPEED = 0.6  # m/s downward: falling


def _quat_m(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose(objects):
    """The battery's pose in the tube frame: (origin, rotation)."""
    b = objects["battery"]
    p = np.asarray(b.pos, float) - np.array([0.0, 0.0, TOP]) - SEAT
    return R_TUBE.T @ p, R_TUBE.T @ _quat_m(np.asarray(b.quat_xyzw, float))


def measures(objects) -> dict:
    """How far the battery is from home: along the tube and sideways at its bottom end, how far its bottom and top ends
    stand off the tube, and its tilt off the tube."""
    t, r = pose(objects)
    bottom = t + r @ np.array([0.0, 0.0, 0.0])
    top = t + r @ np.array([0.0, BL, 0.0])
    lateral_top = top[0]
    tilt = math.atan2(float(np.hypot(r[0, 1], r[2, 1])), float(r[1, 1]))
    return {"axial": float(bottom[1]), "lateral": float(bottom[0]), "bottom_off": float(bottom[2]), "top_off": float(top[2]),
            "lateral_top": float(lateral_top), "tilt": tilt, "roll": float(math.atan2(r[2, 0], r[0, 0]))}


def bottom_home(m) -> bool:
    """The contact block is in the lower holder's pocket: the bottom end within 3 mm of home along the tube and 4 mm
    sideways, on its seat (within 4 mm), the battery square to the tube (within 5 degrees of roll) and tilted off it no
    more than the pocket allows."""
    return bool(abs(m["axial"]) <= SEAT_AXIAL and abs(m["lateral"]) <= SEAT_LATERAL and -0.002 <= m["bottom_off"] <= 0.004
                and abs(m["roll"]) <= math.radians(5.0) and m["tilt"] <= SEAT_TILT)


def swung_in(m) -> bool:
    return bool(bottom_home(m) and m["top_off"] <= SWUNG_IN and abs(m["lateral_top"]) <= SEAT_LATERAL)


def pressed_home(objects, m) -> bool:
    return bool(swung_in(m) and m["top_off"] <= LATCH_TOP and objects["bike"].q["plunger"] <= -PLUNGER_TRAVEL + 0.001)


_last_com = [None]


def com_z(objects) -> float:
    b = objects["battery"]
    return float((np.asarray(b.pos, float) + _quat_m(np.asarray(b.quat_xyzw, float)) @ COM)[2] - TOP)


def _watch(objects, m):
    """A fall: the battery's centre of mass dropping faster than 0.6 m/s (a free fall of 2 cm) between two control
    steps (20 Hz), before it has latched."""
    z = com_z(objects)
    if _last_com[0] is not None and (_last_com[0] - z) / 0.05 > DROP_SPEED and not _latched[0]:
        _dropped[0] = True
    _last_com[0] = z
    if not _latched[0] and pressed_home(objects, m):
        _latched[0] = True
        ebkit.set_flag("latched")


def _trace(objects, m, s) -> None:
    global _calls
    _calls += 1
    if _TRACE and _calls % 4 == 0:
        b = objects["battery"]
        with open(_TRACE, "a") as f:
            f.write(json.dumps({
                "call": _calls, "score": s, "latched": _latched[0], "dropped": _dropped[0],
                "axial_mm": round(m["axial"] * 1000, 1), "lateral_mm": round(m["lateral"] * 1000, 1),
                "bottom_off_mm": round(m["bottom_off"] * 1000, 1), "top_off_mm": round(m["top_off"] * 1000, 1),
                "tilt_deg": round(math.degrees(m["tilt"]), 2), "roll_deg": round(math.degrees(m["roll"]), 2),
                "pos": [round(float(v), 3) for v in (np.asarray(b.pos) - np.array([0, 0, TOP]))],
                "vz": round(float(b.lin_vel[2]), 3),
                "plunger_mm": round(objects["bike"].q["plunger"] * 1000, 2), "bolt_mm": round(objects["bike"].q["bolt"] * 1000, 2),
                "com_z": round(com_z(objects), 4),
                "gap_mm": {k: round(hand_gap(objects, h) * 1000) for k, h in objects["bike"].hands.items()},
                "hands": {k: {"tcp": [round(float(x), 3) for x in (np.asarray(h["tcp"]) - np.array([0, 0, TOP]))],
                              "open": round(h["open"], 3)} for k, h in objects["bike"].hands.items()},
            }) + "\n")


def score(objects):
    """0 if the battery fell. Otherwise, from where it is at the end: 0.25 with its bottom end home in the lower holder,
    0.5 with its top end also swung to within 10 mm of home, 1.0 once it has been pressed home and the latch has
    clicked (and it is still home)."""
    m = measures(objects)
    _watch(objects, m)
    if _dropped[0]:
        s = 0.0
    elif _latched[0] and swung_in(m):
        s = 1.0
    elif swung_in(m):
        s = 0.5
    elif bottom_home(m):
        s = 0.25
    else:
        s = 0.0
    _trace(objects, m, s)
    return s


CLEAR_M = 0.05  # let go: both tool points at least this far from the battery


def hand_gap(objects, hand) -> float:
    """A gripper's tool point to the battery's surface (0 inside it)."""
    b = objects["battery"]
    r = _quat_m(np.asarray(b.quat_xyzw, float))
    p = r.T @ (np.asarray(hand["tcp"], float) - np.asarray(b.pos, float))
    lo, hi = np.array([-BW / 2, 0.0, 0.0]), np.array([BW / 2, BL, BD])
    return float(np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0)))


def let_go(objects) -> bool:
    return all(hand_gap(objects, h) >= CLEAR_M for h in objects["bike"].hands.values())


def success(objects):
    """The episode ends early once the battery has clicked home and both hands are off it (held for 1 s); the score
    does not need the hands off."""
    return bool(score(objects) == 1.0 and let_go(objects))


def distance(objects):
    return 0.0 if score(objects) == 1.0 else 1.0


# -- scripted demo -------------------------------------------------------------------------------------------------------
# A waypoint's ``yaw`` may be a full tool rotation (nine numbers, row by row: columns closing direction, its cross,
# approach direction) instead of a straight-down grip's turn about the vertical.
_down_grip = _demo.grip_rotation


def _grip_rotation(yaw=0.0):
    return np.array(yaw, float).reshape(3, 3) if np.ndim(yaw) else _down_grip(yaw)


_demo.grip_rotation = _grip_rotation


def _rest(side, t, finger, label) -> Waypoint:
    p, r = ik.tcp_pose(np.array(READY, float), (0.0, 0.305 if side == "left" else -0.305, 0.75))
    return Waypoint(t, tuple(float(v) for v in p), finger, label, tuple(r.reshape(-1)))


def _rx(a):
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def _ry(a):
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def _rot_between(r0, r1, s):
    return ik.slerp_matrix(r0, r1, s)


# Each hand holds the battery across its depth (top face and back face), near the arm-side flank, so the lower finger
# stays clear of the down tube: the tool point 30 mm in from the flank's centre line, the approach along the battery's
# width (its X), the fingers closing along its depth (its Z). The grips sit either side of its centre of mass.
GRIP_IN = _env("GRIP_IN", 0.030)
GRIPS = {"right": np.array([-GRIP_IN, 0.080, BD / 2]), "left": np.array([-GRIP_IN, 0.258, BD / 2])}
TOOL_IN_BAT = {"right": np.c_[[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],  # columns: closing (Z), cross, approach (X)
               "left": np.c_[[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]}
COM = np.array([0.0, BL / 2, BD / 2])  # the battery's centre of mass, in its own frame
TILT_IN = math.radians(_env("TILT_IN", 5.0))  # tilt off the tube while the contact block goes into the pocket
ENTRY = _env("ENTRY", 0.030)  # the contact block starts this far up the tube from home before sliding in
HOVER = _env("HOVER", 0.012)  # ... and its seat this far off the tube
SINK = _env("SINK", 0.003)  # while it slides in, its seat is commanded this far into the holder's plate, so it rests on it
PUSH_N = _env("PUSH_N", 22.0)  # each hand's push along the battery's back-face normal while pressing it home (N)
LIFT_C = _env("LIFT_C", 0.12)  # the battery's centre is lifted to this height and rolled there ...
ROLL_LOW = _env("ROLL_LOW", 0.8)  # ... this much of the way (the rest on the way up to the frame) ...
ROLL_X = _env("ROLL_X", 0.23)  # ... its centre at this x
BACK_X = _env("BACK_X", 0.10)  # it is raised this far in front of the down tube, then moved over it
LET_GO = -0.0445  # fingers 89 mm apart: 4 mm clear of the 81 mm battery each side
# The scripted demo plays joint targets through the arms' PD position control (stiffness 80 N m/rad on joints 1-3,
# 10 on joints 4-6); the arms' own weight is compensated, a load is not. So that each arm still holds its share of the
# battery where planned, every target is moved by the joint deflection that share will cause (K^-1 J^T w).
K_JOINT = np.array([80.0, 80.0, 80.0, 10.0, 10.0, 10.0])
COMP = _env("COMP", 1.0)  # 0: no load compensation


def _bat_world(t_tube, r_tube):
    """Battery pose (bench coordinates, z above the bench) from its pose in the tube frame."""
    return SEAT + R_TUBE @ t_tube, R_TUBE @ r_tube


def _tilted(tilt, axial=0.0, off=0.0):
    """The battery tilted ``tilt`` off the tube about its bottom back corner, slid ``axial`` up the tube, ``off`` off it."""
    return np.array([0.0, axial, off]), _rx(tilt)


def _jacobian(q, base):
    p, r = ik.tcp_pose(q, base)
    jac = np.zeros((6, 6))
    for i in range(6):
        dq = q.copy()
        dq[i] += 1e-6
        p2, r2 = ik.tcp_pose(dq, base)
        jac[:3, i] = (p2 - p) / 1e-6
        jac[3:, i] = ik._rot_error(r2, r) / 1e-6
    return jac


def _compensated(plan, base):
    """plan: (duration, tool position, tool rotation, finger, label, wrench at the tool point (force, moment)) per step.
    Returns waypoints whose targets are pre-deflected against each step's wrench."""
    q = np.array(READY, float)
    out = []
    for dur, pos, rot, finger, label, wrench in plan:
        q, _, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q, base_pos=base, iters=120)
        if wrench is not None and COMP:
            dq = (_jacobian(q, base).T @ np.r_[wrench[0], wrench[1]]) / K_JOINT
            pos, rot = ik.tcp_pose(q - COMP * dq, base)
        out.append(Waypoint(dur, tuple(float(v) for v in pos), finger, label, tuple(np.asarray(rot).reshape(-1))))
    return out


def demo():
    """Both hands take the battery from the bench across its depth, lift it and roll it a quarter turn so its back faces
    the down tube, bring its bottom end over the lower holder, tilted 5 degrees, set it on the holder's plate and slide
    the contact block down into the pocket, swing the battery down about it onto the upper holder and press until the
    latch clicks, then let go."""
    c, o = FINGER_CLOSED, FINGER_OPEN
    g = 9.81 * BAT_KG
    # battery key poses: (duration, position, rotation, share of its weight each hand carries, push, label)
    keys = []
    p0, r0 = START_POS.copy(), START_R.copy()
    keys.append((0.0, p0, r0, 0.0, 0.0, "start"))
    p1 = p0 + np.array([0.0, 0.0, LIFT_C - (p0 + r0 @ COM)[2]])
    keys.append((2.0, p1, r0, 0.5, 0.0, "lift it"))
    ta, ra = _tilted(TILT_IN + math.radians(4.0), axial=ENTRY, off=HOVER + 0.03)
    pa, rwa = _bat_world(ta, ra)
    rm = _rot_between(r0, np.eye(3), ROLL_LOW)  # a roll about its long axis only (level), so its ends stay off the bench
    pm = np.array([ROLL_X, 0.0, LIFT_C]) - rm @ COM
    keys.append((3.0, pm, rm, 0.5, 0.0, "roll it most of a quarter turn"))
    keys.append((3.0, pa - np.array([BACK_X, 0.0, 0.0]), rwa, 0.5, 0.0, "up beside the down tube"))
    keys.append((2.0, pa, rwa, 0.5, 0.0, "over the lower holder"))
    pb, rwb = _bat_world(*_tilted(TILT_IN, axial=ENTRY, off=HOVER))
    keys.append((2.0, pb, rwb, 0.5, 0.0, "bottom end over the lower holder's plate"))
    pb2, rwb2 = _bat_world(*_tilted(TILT_IN, axial=ENTRY, off=-SINK))
    keys.append((1.5, pb2, rwb2, 0.35, 0.0, "set it on the plate"))
    pc, rwc = _bat_world(*_tilted(TILT_IN, axial=-0.003, off=-SINK))
    keys.append((2.0, pc, rwc, 0.35, 0.0, "slide the contact block down into the pocket"))
    pd, rwd = _bat_world(*_tilted(0.0, axial=-0.002, off=-SINK))
    keys.append((2.5, pd, rwd, 0.2, 0.0, "swing it down onto the upper holder"))
    keys.append((1.5, pd, rwd, 0.0, PUSH_N, "press until the latch clicks"))
    keys.append((1.0, pd, rwd, 0.0, PUSH_N, "hold"))
    keys.append((1.0, pd, rwd, 0.0, 0.0, "ease off"))

    def tool(side, p, r):
        return p + r @ GRIPS[side] + np.array([0.0, 0.0, TOP]), r @ TOOL_IN_BAT[side]

    def wrench(side, p, r, share, push):
        """What the battery does to the hand: its share of the weight (at the centre of mass) and, while pressing, the
        battery pushing back along its back-face normal."""
        tp, _ = tool(side, p, r)
        f = np.array([0.0, 0.0, -share * g]) + push * (r @ np.array([0.0, 0.0, 1.0]))
        com = p + r @ COM + np.array([0.0, 0.0, TOP])
        m = np.cross(com - tp, np.array([0.0, 0.0, -share * g]))
        return f, m

    plans = {}
    for side in ("left", "right"):
        base = (0.0, 0.305 if side == "left" else -0.305, 0.75)
        pre_p, pre_r = tool(side, p0, r0)
        approach = pre_r[:, 2]
        steps = [(3.0, pre_p - 0.07 * approach, pre_r, o, "over the battery", None),
                 (1.5, pre_p, pre_r, o, "fingers either side", None),
                 (1.0, pre_p, pre_r, c, "grip", None)]
        prev = keys[0]
        for dur, p, r, share, push, label in keys[1:]:
            n = max(1, int(round(dur / 0.1)))
            for i in range(1, n + 1):
                s = ik.smoothstep(i / n)
                pi = prev[1] + s * (p - prev[1])
                ri = _rot_between(prev[2], r, s)
                sh = prev[3] + s * (share - prev[3])
                pu = prev[4] + s * (push - prev[4])
                tp, tr = tool(side, pi, ri)
                steps.append((dur / n, tp, tr, c, label, wrench(side, pi, ri, sh, pu)))
            prev = (dur, p, r, share, push, label)
        # let go: open part way, back off along the approach, then away
        tp, tr = tool(side, prev[1], prev[2])
        ap = tr[:, 2]
        steps += [(1.0, tp, tr, LET_GO, "let go", None),
                  (1.5, tp - 0.08 * ap, tr, LET_GO, "back off", None)]
        plans[side] = _compensated(steps, base) + [_rest(side, 3.0, o, "away"), _rest(side, 2.0, o, "wait")]
    return plans
