"""On the station table, centred between the arms: a shallow open aluminium tray, 12 cm wide (along x) and 16 cm long
(along y), with a 2 mm lip and a grey lug on each end to hold it by, about 13 to 25 cm out from the arm bases. Six loose
steel balls, 2 cm across, lie in it in two rows of three, free to roll. Further out, 28.5 to 43.5 cm from the arm bases,
is a fixed dark fixture: a plate 3 cm high with rails around a pocket that fits the tray, gaps in the end rails for
the lugs.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Put the tray in the fixture without spilling any of the steel balls." \\
           --scorer task_score -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from traykit import compound  # noqa: E402  (importing traykit hooks yam_arena so the tray is simulated)
from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, Sphere  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

SILVER, GREY, DARK, STEEL = (0.75, 0.77, 0.80), (0.35, 0.37, 0.40), (0.16, 0.17, 0.20), (0.90, 0.91, 0.93)

# -- tray: frame here = centre of its underside ------------------------------------------------------------------------

TRAY_X, TRAY_Y = 0.12, 0.16  # outer size of the pan
FLOOR_T = 0.003
LIP_H, LIP_T = 0.002, 0.004  # the lip rises 2 mm above the floor: a 2 cm ball rolls over it if it hits it fast enough
LUG_X, LUG_END, LUG_H = 0.016, 0.14, 0.030  # grip lugs along y, from the pan's end out to |y| = LUG_END
TRAY_MASS = 0.25


def _tray_boxes() -> list:
    wall_z = FLOOR_T + LIP_H / 2
    inner_y = TRAY_Y - 2 * LIP_T
    lug_y = (TRAY_Y / 2 + LUG_END) / 2
    lug_len = LUG_END - TRAY_Y / 2
    return [
        ((0.0, 0.0, FLOOR_T / 2), (TRAY_X, TRAY_Y, FLOOR_T), SILVER),
        (((TRAY_X - LIP_T) / 2, 0.0, wall_z), (LIP_T, TRAY_Y, LIP_H), SILVER),
        ((-(TRAY_X - LIP_T) / 2, 0.0, wall_z), (LIP_T, TRAY_Y, LIP_H), SILVER),
        ((0.0, (TRAY_Y - LIP_T) / 2, wall_z), (TRAY_X - 2 * LIP_T, LIP_T, LIP_H), SILVER),
        ((0.0, -(TRAY_Y - LIP_T) / 2, wall_z), (TRAY_X - 2 * LIP_T, LIP_T, LIP_H), SILVER),
        ((0.0, lug_y, LUG_H / 2), (LUG_X, lug_len, LUG_H), GREY),
        ((0.0, -lug_y, LUG_H / 2), (LUG_X, lug_len, LUG_H), GREY),
    ] if inner_y > 0 else []


START_X, START_Y = 0.19, 0.0  # the tray's centre
REST_GAP = 0.0005
TRAY = compound("tray", pos=(START_X, START_Y, TOP + LUG_H / 2 + REST_GAP), boxes=_tray_boxes(), mass=TRAY_MASS,
                color=SILVER)
TRAY_BOTTOM_TO_CENTRE = LUG_H / 2  # the tray's state.pos is the centre of its bounding box

# -- balls -------------------------------------------------------------------------------------------------------------

BALL_R, BALL_MASS = 0.01, 0.033  # 2 cm steel ball
BALL_X, BALL_Y = (-0.025, 0.025), (-0.045, 0.0, 0.045)  # on the tray, relative to its centre
BALLS = [
    Sphere(f"ball_{i}", pos=(START_X + dx, START_Y + dy, TOP + REST_GAP + FLOOR_T + BALL_R + 0.0005), radius=BALL_R,
           mass=BALL_MASS, color=STEEL)
    for i, (dx, dy) in enumerate((dx, dy) for dx in BALL_X for dy in BALL_Y)
]
BALL_NAMES = tuple(b.name for b in BALLS)

# -- fixture: fixed plain boxes -----------------------------------------------------------------------------------------

FIX_X, FIX_Y = 0.36, 0.0  # the pocket's centre
PLATE_H = 0.030  # the pocket's floor above the table: too high to push the tray up
RAIL_H, RAIL_T = 0.010, 0.010
CLEAR = 0.005  # free play around the pan on every side
POCKET_X, POCKET_Y = TRAY_X + 2 * CLEAR, TRAY_Y + 2 * CLEAR
PLATE_X, PLATE_Y = POCKET_X + 2 * RAIL_T, POCKET_Y + 2 * RAIL_T
LUG_GAP = 0.024  # the gap in each end rail for a lug
STUB_X = (POCKET_X - LUG_GAP) / 2


def _fixture() -> list:
    rail_z = TOP + PLATE_H + RAIL_H / 2
    rail_x = (POCKET_X + RAIL_T) / 2
    rail_y = (POCKET_Y + RAIL_T) / 2
    stub_x = (LUG_GAP + STUB_X) / 2
    parts = [
        Box("fixture_plate", pos=(FIX_X, FIX_Y, TOP + PLATE_H / 2), size=(PLATE_X, PLATE_Y, PLATE_H), color=DARK),
        Box("fixture_far_rail", pos=(FIX_X + rail_x, FIX_Y, rail_z), size=(RAIL_T, PLATE_Y, RAIL_H), color=DARK),
        Box("fixture_near_rail", pos=(FIX_X - rail_x, FIX_Y, rail_z), size=(RAIL_T, PLATE_Y, RAIL_H), color=DARK),
    ]
    for label, sy in (("left", 1.0), ("right", -1.0)):
        for side, sx in (("far", 1.0), ("near", -1.0)):
            parts.append(Box(f"fixture_{label}_{side}_stub", pos=(FIX_X + sx * stub_x, FIX_Y + sy * rail_y, rail_z),
                             size=(STUB_X, RAIL_T, RAIL_H), color=DARK))
    return parts


OBJECTS = [TRAY, *BALLS, *_fixture()]

# -- scoring -----------------------------------------------------------------------------------------------------------

SEAT_XY = 0.010  # m the tray's centre may be from the pocket's centre; the rails allow only CLEAR
SEAT_Z = 0.003  # m the tray's underside may be above the pocket's floor: down in the pocket, not up on a 1 cm rail
IN_TRAY_HIGH = 0.005  # m a ball's underside may be above the tray's floor: not held up, not on another ball
IN_TRAY_LOW = 0.005  # m it may be below: contact offsets, never a ball that has left the tray
STILL = 0.02  # m/s


def _rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tray_seated(tray) -> bool:
    """The tray lies in the pocket: its centre within 1 cm of the pocket's, its underside within 3 mm of the floor."""
    offset = np.hypot(tray.pos[0] - FIX_X, tray.pos[1] - FIX_Y)
    bottom_above_floor = tray.pos[2] - TRAY_BOTTOM_TO_CENTRE - (TOP + PLATE_H)
    return bool(offset <= SEAT_XY and abs(bottom_above_floor) <= SEAT_Z)


def ball_in_tray(ball, tray) -> bool:
    """The ball lies on the tray's floor, inside its lip."""
    local = _rotation(tray.quat_xyzw).T @ (np.asarray(ball.pos) - np.asarray(tray.pos))
    height = local[2] + TRAY_BOTTOM_TO_CENTRE - FLOOR_T - BALL_R  # ball's underside above the floor
    return bool(abs(local[0]) <= TRAY_X / 2 - LIP_T and abs(local[1]) <= TRAY_Y / 2 - LIP_T
                and -IN_TRAY_LOW <= height <= IN_TRAY_HIGH)


def balls_in_tray(objects) -> int:
    return sum(ball_in_tray(objects[n], objects["tray"]) for n in BALL_NAMES)


def score(objects):
    """With the tray seated in the fixture: 1.0 with all six balls in it, 0.75 with five, 0.5 with three or four,
    0.25 with one or two. 0 with none, or if the tray isn't seated."""
    if not tray_seated(objects["tray"]):
        return 0.0
    count = balls_in_tray(objects)
    return 1.0 if count == 6 else 0.75 if count == 5 else 0.5 if count >= 3 else 0.25 if count >= 1 else 0.0


def success(objects):
    """Tray seated with all six balls in it, and everything still (held for SUCCESS_HOLD_S)."""
    still = all(np.linalg.norm(objects[n].lin_vel) < STILL for n in ("tray", *BALL_NAMES))
    return bool(score(objects) == 1.0 and still)


def distance(objects):
    """Metres from the tray's centre to the pocket's centre, flat."""
    return float(np.hypot(objects["tray"].pos[0] - FIX_X, objects["tray"].pos[1] - FIX_Y))


# -- scripted demo -----------------------------------------------------------------------------------------------------

GRIP_Y = 0.12  # on each lug: the pads, 4 cm wide, reach from 10 to 14 cm, outside the fixture's end rails
GRIP_ABOVE_BOTTOM = 0.020  # TCP above the tray's underside: the pad tips stop 7 mm above it
GRIP_OPEN = -0.022  # finger target (m): pads 44 mm apart around the 16 mm lug
LIFT = PLATE_H + RAIL_H + 0.015  # the tray's underside clears the rails by 1.5 cm
HOVER = 0.10
CARRY_S = 5.0  # slow: at this pace the balls barely roll
FAR_LAG = 0.0025  # measured in Isaac: at the fixture's reach the arms stop 2.4 mm short of the commanded x


def _arm(sign: float, carry_s: float = CARRY_S) -> list[Waypoint]:
    """One arm's half of the carry; both arms run the same timing. ``sign`` is +1 for the left arm's lug."""
    y = sign * GRIP_Y
    yaw = -sign * math.pi / 2  # fingers close along x, across the lug
    z = TOP + GRIP_ABOVE_BOTTOM
    lifted = z + LIFT
    down = z + PLATE_H + 0.002  # let go 2 mm above the pocket's floor
    far = FIX_X + FAR_LAG
    return [
        Waypoint(2.0, (START_X, y, z + HOVER), GRIP_OPEN, "above the lug", yaw),
        Waypoint(1.5, (START_X, y, z), GRIP_OPEN, "down around it", yaw),
        Waypoint(1.0, (START_X, y, z), FINGER_CLOSED, "grip", yaw),
        Waypoint(3.0, (START_X, y, lifted), FINGER_CLOSED, "lift gently", yaw),
        Waypoint(carry_s, (far, y, lifted), FINGER_CLOSED, "carry", yaw),
        Waypoint(1.0, (far, y, lifted), FINGER_CLOSED, "let the balls settle", yaw),
        Waypoint(3.0, (far, y, down), FINGER_CLOSED, "lower into the pocket", yaw),
        Waypoint(1.0, (far, y, down), GRIP_OPEN, "let go", yaw),
        Waypoint(1.5, (far, y, z + HOVER), GRIP_OPEN, "up", yaw),
    ]


def demo():
    """Both arms hold a lug each and carry the tray level and slowly into the fixture."""
    return {"left": _arm(1.0), "right": _arm(-1.0)}
