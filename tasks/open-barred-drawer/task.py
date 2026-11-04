"""On the station table, centred between the arms: a light-grey desktop drawer cabinet, 30 cm wide, 24 cm deep and
21.5 cm tall, its front 30 cm out from the arm bases and facing them. It has three drawers with blue fronts and black
bar handles; the top drawer's handle is 18.75 cm above the table. A red bar, 44 cm long and 1.5 cm square, lies
across the drawer fronts in two dark-grey closed sleeves on the cabinet's front corners, 2 cm in front of the top
drawer, so the top drawer can only open about 2 cm until the bar is slid out sideways. The cabinet is fixed
to the table; the drawers slide out up to 20 cm; the bar is loose.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --instruction "Open the top drawer." \\
           --scorer task_score -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from drawerkit import Drawer, cabinet  # noqa: E402  (importing drawerkit hooks yam_arena so the cabinet is simulated)
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 0.5
SCORE_OVER = "end"

# -- geometry: cabinet frame = front face (x 0, +x into the cabinet), centred in y, z 0 on the table top ------------

CAB_X, CAB_Y = 0.30, 0.0  # world position of the cabinet frame (front face, centre line)
W, D, H, T = 0.30, 0.24, 0.215, 0.01  # outer width (y), depth (x), height (z), wall thickness

GREY, BLUE, BLACK, DARK, RED = (0.78, 0.78, 0.76), (0.20, 0.40, 0.85), (0.12, 0.12, 0.12), (0.30, 0.30, 0.32), (
    0.85, 0.10, 0.10)

FRONT_H, FRONT_T, FRONT_W = 0.060, 0.012, W - 2 * T - 0.002  # drawer fronts: height, thickness, width
FRONT_Z = (0.0125, 0.0775, 0.1425)  # bottom of each front above the table: bottom, middle, top drawer
TRAVEL = 0.20

HANDLE_LEN, HANDLE_T, HANDLE_H, STEM = 0.07, 0.012, 0.020, 0.035  # grip bar along y; stems stand it off the front

BAR_SIZE = (0.015, 0.44, 0.015)  # square tube: x, y (length), z
BAR_GAP = 0.020  # between the bar's back face and the top drawer's front
BAR_BOTTOM = 0.1475  # above the table, i.e. 5 mm above the top drawer front's bottom edge
BRACKET_Y = (0.150, 0.170)  # |y| span of each bracket, outside the side walls
LIP_H = 0.020  # the lips rise this far above the floor, leaving 5 mm over the bar: it cannot be lifted out
LIP_T = 0.008
CAP_T = 0.005  # the sleeve's top, closing the lips: the bar's only way out is along its length


def _carcass() -> list:
    boxes = [
        ((D / 2, W / 2 - T / 2, H / 2), (D, T, H), GREY),  # left side
        ((D / 2, -(W / 2 - T / 2), H / 2), (D, T, H), GREY),  # right side
        ((D / 2, 0.0, H - T / 2), (D, W, T), GREY),  # top
        ((D / 2, 0.0, T / 2), (D, W, T), GREY),  # bottom
        ((D - T / 2, 0.0, H / 2), (T, W - 2 * T, H - 2 * T), GREY),  # back
    ]
    for z in (FRONT_Z[1] - 0.0025, FRONT_Z[2] - 0.0025):  # runners between the drawers
        boxes.append(((D / 2, 0.0, z), (D - T, W - 2 * T, 0.004), GREY))
    bar_x0 = -BAR_GAP - BAR_SIZE[0]  # bar's front face in the cabinet frame
    floor_x0 = bar_x0 - 0.002 - LIP_T  # brackets reach this far in front of the cabinet
    for s in (1.0, -1.0):
        y = s * sum(BRACKET_Y) / 2
        wy = BRACKET_Y[1] - BRACKET_Y[0]
        floor_z = BAR_BOTTOM - 0.005
        boxes += [
            ((floor_x0 / 2, y, floor_z), (-floor_x0, wy, 0.010), DARK),  # floor the bar rests on
            ((floor_x0 + LIP_T / 2, y, BAR_BOTTOM + LIP_H / 2 - 0.005), (LIP_T, wy, LIP_H + 0.010), DARK),  # front lip
            ((-0.011, y, BAR_BOTTOM + LIP_H / 2 - 0.005), (LIP_T, wy, LIP_H + 0.010), DARK),  # back lip
            ((floor_x0 / 2, y, BAR_BOTTOM + LIP_H + CAP_T / 2), (-floor_x0, wy, CAP_T), DARK),  # top
        ]
    return boxes


def _drawer(name: str, z0: float, handle_z: float) -> Drawer:
    """A drawer whose frame sits at its front's bottom edge, on the cabinet's front plane."""
    inner_w = FRONT_W - 0.008
    depth = D - T - FRONT_T - 0.008
    side_h = FRONT_H - 0.012
    boxes = [
        ((FRONT_T / 2, 0.0, FRONT_H / 2), (FRONT_T, FRONT_W, FRONT_H), BLUE),  # front
        ((FRONT_T + depth / 2, 0.0, 0.007), (depth, inner_w, 0.004), BLUE),  # bottom
        ((FRONT_T + depth / 2, inner_w / 2 - 0.002, side_h / 2 + 0.005), (depth, 0.004, side_h), BLUE),  # sides
        ((FRONT_T + depth / 2, -(inner_w / 2 - 0.002), side_h / 2 + 0.005), (depth, 0.004, side_h), BLUE),
        ((FRONT_T + depth - 0.002, 0.0, side_h / 2 + 0.005), (0.004, inner_w, side_h), BLUE),  # back
        ((-STEM / 2, 0.025, handle_z), (STEM, 0.008, 0.008), BLACK),  # handle stems
        ((-STEM / 2, -0.025, handle_z), (STEM, 0.008, 0.008), BLACK),
        ((-STEM - HANDLE_T / 2, 0.0, handle_z), (HANDLE_T, HANDLE_LEN, HANDLE_H), BLACK),  # handle grip
    ]
    return Drawer(name, closed=(0.0, 0.0, z0), boxes=tuple(boxes), travel=TRAVEL, damping=5.0, mass=0.4)


TOP_HANDLE_Z = 0.045  # in the top drawer's frame: upper part of its front, clear above the bar
CABINET = cabinet(
    "cabinet",
    pos=(CAB_X, CAB_Y, TOP),
    boxes=_carcass(),
    drawers=[
        _drawer("bottom_drawer", FRONT_Z[0], FRONT_H / 2),
        _drawer("middle_drawer", FRONT_Z[1], FRONT_H / 2),
        _drawer("top_drawer", FRONT_Z[2], TOP_HANDLE_Z),
    ],
    color=GREY,
)
BAR_POS = (CAB_X - BAR_GAP - BAR_SIZE[0] / 2, CAB_Y, TOP + BAR_BOTTOM + BAR_SIZE[2] / 2)
BAR = Box("bar", pos=BAR_POS, size=BAR_SIZE, mass=0.10, color=RED)
OBJECTS = [CABINET, BAR]

# -- scoring --------------------------------------------------------------------------------------------------------

OPEN_GOAL = 0.18  # m the top drawer's front must come out
STILL = 0.02  # m/s


def bracket_slots() -> list[tuple[np.ndarray, np.ndarray]]:
    """World boxes (low, high) of the two sleeve slots: between the lips, from the floor to the sleeve's top."""
    bar_x0 = -BAR_GAP - BAR_SIZE[0]
    x_lo, x_hi = CAB_X + bar_x0 - 0.002, CAB_X - 0.011 - LIP_T / 2
    z_lo, z_hi = TOP + BAR_BOTTOM, TOP + BAR_BOTTOM + LIP_H
    return [(np.array([x_lo, CAB_Y + y0, z_lo]), np.array([x_hi, CAB_Y + y1, z_hi]))
            for y0, y1 in ((BRACKET_Y[0], BRACKET_Y[1]), (-BRACKET_Y[1], -BRACKET_Y[0]))]


def bar_clear(bar) -> bool:
    """True once no part of the bar's bounding box is inside either bracket slot."""
    for low, high in bracket_slots():
        if np.all(bar.low < high - 1e-4) and np.all(bar.high > low + 1e-4):
            return False
    return True


def success(objects):
    """The top drawer is out at least 18 cm and no longer moving (held for SUCCESS_HOLD_S)."""
    drawer = objects["cabinet"]
    return bool(drawer.opening["top_drawer"] >= OPEN_GOAL and abs(drawer.speed["top_drawer"]) < STILL)


def distance(objects):
    """Metres the top drawer still has to come out."""
    return max(0.0, OPEN_GOAL - objects["cabinet"].opening["top_drawer"])


def score(objects):
    """1.0 with the top drawer out 18 cm; else 0.5 once the bar is out of both brackets; else 0."""
    if objects["cabinet"].opening["top_drawer"] >= OPEN_GOAL:
        return 1.0
    return 0.5 if bar_clear(objects["bar"]) else 0.0


# -- scripted demo ----------------------------------------------------------------------------------------------------

BAR_GRIP_Y = 0.205  # the left arm's grip, clear of the left sleeve: pads reach 2 cm either side
BAR_GRIP_OPEN = -0.025  # finger target (m): pads 50 mm apart, enough for the 15 mm bar
BAR_GRIP_Z = BAR_POS[2] + 0.008  # the pad tips reach 13 mm past the TCP: this stops them short of the table
BAR_HOVER = 0.04  # above the grip height; straight-down grips this far from the base run out of wrist above 6 cm
FIRST_SLIDE = 0.16  # the bar's centre ends over the left sleeve, which keeps it level when let go
BAR_OUT = 0.40  # total slide: the bar's near end clears the left sleeve, which ends at y 0.17
HANDLE_X = CAB_X - STEM - HANDLE_T / 2
HANDLE_Z = TOP + FRONT_Z[2] + TOP_HANDLE_Z
GRIP_HANDLE_Z = TOP + 0.18  # straight down, the arm reaches the handle only this low: 7.5 mm below its centre
PULL = 0.19


def demo():
    """The left arm slides the bar out in two grips and lays it down; the right arm then opens the drawer."""
    x, y, z = BAR_POS[0], CAB_Y + BAR_GRIP_Y, BAR_GRIP_Z
    across = math.pi / 2  # fingers close along x, across the bar
    after_first = y + FIRST_SLIDE
    out = y + BAR_OUT - FIRST_SLIDE
    on_table = TOP + BAR_SIZE[2] / 2 + 0.008 + 0.003
    left = [
        Waypoint(2.0, (x, y, z + BAR_HOVER), BAR_GRIP_OPEN, "above the bar's end", across),
        Waypoint(1.0, (x, y, z), BAR_GRIP_OPEN, "descend", across),
        Waypoint(0.8, (x, y, z), FINGER_CLOSED, "grip", across),
        Waypoint(3.0, (x, after_first, z), FINGER_CLOSED, "slide the bar out of the right sleeve", across),
        Waypoint(0.6, (x, after_first, z), BAR_GRIP_OPEN, "let go", across),
        Waypoint(0.8, (x, after_first, z + BAR_HOVER), BAR_GRIP_OPEN, "up", across),
        Waypoint(1.5, (x, y, z + BAR_HOVER), BAR_GRIP_OPEN, "back over the bar's middle", across),
        Waypoint(0.8, (x, y, z), BAR_GRIP_OPEN, "descend", across),
        Waypoint(0.8, (x, y, z), FINGER_CLOSED, "grip", across),
        Waypoint(4.0, (x, out, z), FINGER_CLOSED, "slide the bar out of the left sleeve", across),
        Waypoint(2.0, (x, out, on_table), FINGER_CLOSED, "lower onto the table", across),
        Waypoint(0.6, (x, out, on_table), BAR_GRIP_OPEN, "release", across),
        Waypoint(1.0, (x, out, TOP + 0.12), BAR_GRIP_OPEN, "up", across),
    ]
    bar_done = sum(w.duration for w in left)
    front = (HANDLE_X - 0.10, CAB_Y, GRIP_HANDLE_Z)
    at_handle = (HANDLE_X, CAB_Y, GRIP_HANDLE_Z)
    pulled = (HANDLE_X - PULL, CAB_Y, GRIP_HANDLE_Z)
    right = [
        Waypoint(2.0, front, FINGER_OPEN, "in front of the top handle"),
        Waypoint(bar_done - 2.0, front, FINGER_OPEN, "wait for the bar"),
        Waypoint(1.5, at_handle, FINGER_OPEN, "in around the handle's ends"),
        Waypoint(0.8, at_handle, FINGER_CLOSED, "grip the handle's ends"),
        Waypoint(5.0, pulled, FINGER_CLOSED, "pull the drawer out"),
        Waypoint(1.5, pulled, FINGER_CLOSED, "hold"),
    ]
    return {"left": left, "right": right}
