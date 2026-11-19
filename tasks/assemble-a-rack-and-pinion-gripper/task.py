"""Assemble a rack-and-pinion parallel gripper so both fingers sit evenly about its centre.

A 3D-printed parallel-jaw gripper lies on the bench: a housing with two channels, one either side of a pin pinion
(eight 3 mm steel dowel pins on a 24 mm pitch circle, free to turn), and two racks that each carry a finger plate. The
white rack is already in the near channel, one tooth (9.42 mm) open, and the pinion has turned with it. The orange
rack lies loose beside the housing. Seated on the pinion at the right tooth, the two fingers sit at equal distances
from the housing's centre; one tooth off, they sit 4.71 mm off-centre for good. The robot seats the orange rack, then
opens the jaws by moving the white rack alone, which drives the orange rack through the pinion.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 \\
           -E control_hz=20 -E cam_width=1280 -E cam_height=704 --max-steps 1200 --instruction "<instruction.txt>"
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import gripkit  # noqa: E402  (importing gripkit hooks yam_arena so the fixtures are simulated)
import numpy as np  # noqa: E402
from gripkit import Slider, fixture  # noqa: E402
from yam_arena import ik  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 4.0  # the finished gripper, released and clear, held 4 s
SCORE_OVER = "end"
gripkit.CONTACT_OFFSET = 0.0005  # sub-millimetre clearances: contacts start 0.5 mm out


def _env(name, default):
    return float(os.environ.get(name, default))


GREY, WHITE, ORANGE, STEEL, DARK = (0.16, 0.17, 0.19), (0.93, 0.93, 0.91), (0.95, 0.45, 0.08), (0.62, 0.64, 0.67), (0.20, 0.21, 0.23)
VB = {"collide": False}

# -- geometry (metres). Housing frame: origin at the middle of the housing's underside, y along its length (world y),
#    x across it (world x), z up. --------------------------------------------------------------------------------------
XH, YH = _env("XH", 0.24), 0.0  # the housing's middle on the bench (world)
HL, HW_OUT = 0.160, 0.052  # housing floor: 160 x 52 mm
FT = 0.010  # floor top: 6 mm steel base plate + 4 mm printed floor
WALL_IN, WALL_T, WALL_H = 0.0225, 0.0035, 0.012  # walls: inner faces 45 mm apart, 3.5 mm thick, 12 mm tall
RAIL_H = 0.005  # each rack rides on a 5 mm rail under its body
RB = FT + RAIL_H  # rack underside
R = 0.012  # pinion pitch radius (pins on a 24 mm circle)
PIN_D, PIN_H = 0.003, 0.011  # 3 x 16 mm dowel pins, 12.5 mm standing (rounded end) and 3.5 mm pressed into ...
DISC_D, DISC_T = 0.0272, 0.004  # ... a 27.2 mm disc, 4 mm thick, resting on the floor
P = 2 * math.pi * R / 8  # circular pitch, 9.42 mm (module 3)
GAP = _env("GAP", 0.0034)  # rack tooth gap: 3.4 mm round a 3 mm pin (0.4 mm backlash per rack)
TW = P - GAP  # tooth width along the rack
T_IN, T_OUT = R - 0.0010, R + 0.0018  # teeth run from 1 mm inside the pitch line to the rack body face (1.8 mm out)
BODY_W = 0.008  # rack body, 8 mm wide, so 0.7 mm clear of the wall
RL, RH = 0.088, 0.012  # rack length and height
CLOSED_C = 0.004  # a closed rack's middle sits 4 mm from the pinion axis along y (near: -y, far: +y)
START_OPEN = P  # the white rack starts one tooth open
STOP_Y = RL / 2 - CLOSED_C  # stop blocks at the closed racks' inner ends (32 mm from the axis)
BOSS = (0.007, 0.006, 0.0195)  # the finger-mount boss on each rack (real fingers bolt to it), 7 x 6 x 19.5 mm
PLATE = (0.0075, 0.006, 0.030)  # finger plate at each rack's outer end, as wide as the rack body
HOUSING_MASS, RACK_MASS, PINION_MASS = 0.46, 0.013, 0.010


def _rack_boxes(side, colour):
    """A rack in its own frame: origin on the pitch line at the rack's middle, at its underside. side = +1 for the far
    rack (on the +x side of the pinion, opening toward +y), -1 for the near rack. Gaps sit where the pins do when the
    rack is closed: at housing y = 0, i.e. local y = -side * CLOSED_C, and every pitch from there."""
    s = side
    bx = (T_OUT - R) + BODY_W / 2  # body centre from the pitch line
    out = [((s * bx, 0.0, RH / 2), (BODY_W, RL, RH), colour)]
    g0 = -s * CLOSED_C
    k0 = math.ceil((-RL / 2 - g0) / P)
    k1 = math.floor((RL / 2 - g0) / P)
    for k in range(k0 - 1, k1 + 1):  # teeth between consecutive gaps
        yc = g0 + (k + 0.5) * P
        if abs(yc) > RL / 2 - TW / 2 + 1e-9:
            continue
        xa = (T_IN + 0.0006 + T_OUT) / 2 - R
        out += [((s * xa, yc, 0.0015 + (RH - 0.0015) / 2), (T_OUT - T_IN - 0.0006, TW, RH - 0.0015), colour),  # tooth
                ((s * (T_IN + 0.0003 - R), yc, 0.0015 + (RH - 0.0015) / 2), (0.0006, TW - 0.0012, RH - 0.0015), colour),  # rounded crest
                ((s * ((T_IN + T_OUT) / 2 - R), yc, 0.00075), (T_OUT - T_IN, TW - 0.0016, 0.0015), colour)]  # lead-in foot
    out.append(((s * bx, s * (RL / 2 + PLATE[1] / 2), PLATE[2] / 2), PLATE, colour))  # finger plate
    out.append(((s * bx, 0.0, RH + BOSS[2] / 2), BOSS, colour))  # finger-mount boss
    out.append(((s * (bx + BODY_W / 2 - 0.0008), 0.0, 0.0008), (0.0016, RL, 0.0016), colour, {"shape": "cyl", "axis": "Y"}))  # rounded outer foot
    return out


_pins = []
for k in range(8):
    a = k * math.pi / 4
    c = (R * math.cos(a), R * math.sin(a))
    _pins += [((c[0], c[1], DISC_T + PIN_H / 2), (PIN_D, PIN_D, PIN_H), STEEL, {"shape": "cyl", "axis": "Z"}),
              ((c[0], c[1], DISC_T + PIN_H), (PIN_D, PIN_D, PIN_D), STEEL, {"shape": "sphere"})]
PINION = Slider("pinion", (0.0, 0.0, FT), "Z", (
    ((0.0, 0.0, DISC_T / 2), (DISC_D, DISC_D, DISC_T), DARK, {"shape": "cyl", "axis": "Z"}),
    ((0.0, 0.0, 0.007), (0.010, 0.010, 0.014), STEEL, {"shape": "cyl", "axis": "Z"}),  # hub on its shoulder screw
    *_pins), kind="revolute", low=-20.0, high=20.0, stiffness=0.0, damping=0.0004, friction=0.0, mass=PINION_MASS)

_lane = (T_OUT + 0.0002 + WALL_IN) / 2  # rails and stops sit under/against the rack bodies only
_housing = [
    ((0.0, 0.0, FT / 2), (HW_OUT, HL, FT), GREY),  # floor (on a steel base plate)
    ((0.0, 0.0, 0.003), (HW_OUT + 0.0004, HL + 0.0004, 0.006), DARK, VB),  # the steel plate's edge (as seen)
    *[((sx * (WALL_IN + WALL_T / 2), 0.0, FT + WALL_H / 2), (WALL_T, HL - 0.010, WALL_H), GREY) for sx in (-1, 1)],
    *[((sx * (WALL_IN + 0.0006), 0.0, FT + WALL_H), (0.0012, HL - 0.010, 0.0012), GREY, {"shape": "cyl", "axis": "Y"})
      for sx in (-1, 1)],  # rounded inner edge of each wall top (lead-in)
    *[((sx * _lane, 0.0, FT + RAIL_H / 2), (WALL_IN - T_OUT - 0.0002, HL - 0.010, RAIL_H), GREY) for sx in (-1, 1)],
    ((-_lane, STOP_Y + 0.002, RB + 0.0045), (WALL_IN - T_OUT - 0.0004, 0.004, 0.009), STEEL),  # near rack's stop
    ((_lane, -STOP_Y - 0.002, RB + 0.0045), (WALL_IN - T_OUT - 0.0004, 0.004, 0.009), STEEL),  # far rack's stop
    *[((sx * (WALL_IN + 0.0024), 0.0, FT + WALL_H + 0.0001), (0.0016, HL - 0.012, 0.0002), (0.98, 0.85, 0.10), VB)
      for sx in (-1, 1)],  # 1 mm scales along each wall top (yellow tape)
    *[((sx * (WALL_IN + 0.0024), sy * k * 0.010, FT + WALL_H + 0.0002), (0.0016, 0.0008, 0.0002), DARK, VB)
      for sx in (-1, 1) for sy in (-1, 1) for k in range(0, 8)],  # their 10 mm marks
]
HOUSING = fixture("housing", (XH, YH, TOP), _housing, [PINION], fixed=False, base_mass=HOUSING_MASS, friction=0.4)

NEAR0 = (XH - R, YH - CLOSED_C - START_OPEN, TOP + RB)  # the white rack's start: in its channel, one tooth open
FAR0 = (_env("FAR_X", XH + R + 0.0), _env("FAR_Y", 0.14), TOP)  # the orange rack's start: on the bench, left of the housing
NEAR = fixture("near", NEAR0, _rack_boxes(-1, WHITE), [], fixed=False, base_mass=RACK_MASS, friction=0.25)
FAR = fixture("far", FAR0, _rack_boxes(+1, ORANGE), [], fixed=False, base_mass=RACK_MASS, friction=0.25)
OBJECTS = [HOUSING, NEAR, FAR]

# -- scoring ------------------------------------------------------------------------------------------------------------
STEP_S = 0.05
SEAT_X, SEAT_Z, SEAT_TILT, SEAT_YAW = 0.0015, 0.0015, math.radians(3), math.radians(4)
SYM = 0.002  # |d| = |s_near - s_far| / 2 at most 2 mm (one tooth off is 4.71 mm)
OPEN = 0.015  # both fingers at least 15 mm open ...
COUPLED = 0.015  # ... with at least 15 mm of opening in which both racks moved apart while one of them was untouched
NEAR_R = 0.005  # "untouched": no robot contact and no finger pad within 5 mm (tool point 20 mm from the rack's body)
CLEAR = 0.030  # release: every finger at least 30 mm from the housing and racks (tool point 45 mm) ...
STILL = 0.003  # ... and the racks still (under 3 mm/s)

_TRACE = os.environ.get("TASK_TRACE")
_EVERY = int(os.environ.get("TRACE_EVERY", 10))
_H = {"last": None, "prev": None, "coupled": 0.0}


def _reset_record():
    _H.clear()
    _H.update({"last": None, "prev": None, "coupled": 0.0})


gripkit.RESET_HOOKS.append(_reset_record)  # cleared at every environment reset (one record per episode)


def _rotm(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rack_in_housing(objects, name):
    """The rack's origin and rotation in the housing's frame."""
    h, r = objects["housing"], objects[name]
    Rh = _rotm(h.quat_xyzw)
    p = Rh.T @ (np.asarray(r.pos, float) - np.asarray(h.pos, float))
    return p, Rh.T @ _rotm(r.quat_xyzw)


def seated(objects, name) -> bool:
    """The rack lies in its own channel: pitch line within 1.5 mm, underside within 1.5 mm of its rail, within 3 degrees
    of level and 4 of straight."""
    side = 1 if name == "far" else -1
    p, Rr = rack_in_housing(objects, name)
    tilt = math.acos(float(np.clip(Rr[2, 2], -1, 1)))
    yaw = math.atan2(Rr[1, 0], Rr[0, 0])
    return bool(abs(p[0] - side * R) <= SEAT_X and abs(p[2] - RB) <= SEAT_Z and tilt <= SEAT_TILT and abs(yaw) <= SEAT_YAW)


def _tooth_span(side):
    """The first and last tooth centres along the rack (its own y)."""
    g0 = -side * CLOSED_C
    ys = [g0 + (k + 0.5) * P for k in range(-12, 12) if abs(g0 + (k + 0.5) * P) <= RL / 2 - TW / 2 + 1e-9]
    return min(ys), max(ys)


def overlaps(objects, name) -> bool:
    """The pinion's axis lies along the rack's toothed length, between its first and last tooth, so the pin at the
    mesh point has a tooth on each side of it (at least 2 teeth of the rack over the pinion)."""
    side = 1 if name == "far" else -1
    p, _ = rack_in_housing(objects, name)
    lo, hi = _tooth_span(side)
    y_axis = -p[1]  # the pinion's axis (housing y = 0) in the rack's own y
    return bool(lo <= y_axis <= hi)


def meshed(objects, name) -> bool:
    """Seated, over the pinion, and in mesh with it: the pin at the mesh point sits in one of the rack's tooth gaps
    (its centre within the gap's clearance, 0.2 mm, plus 0.5 mm). A rack resting on the pin tips is not seated; a rack
    beyond the pinion does not overlap it."""
    if not (seated(objects, name) and overlaps(objects, name)):
        return False
    side = 1 if name == "far" else -1
    p, _ = rack_in_housing(objects, name)
    th = float(objects["housing"].q["pinion"])
    target = 0.0 if side > 0 else math.pi  # the mesh point: +x for the far rack, -x for the near one
    k = round((target - th) / (math.pi / 4))
    pin_y = R * math.sin(th + k * math.pi / 4)  # the pin nearest the mesh point
    g0 = -side * CLOSED_C
    gap_local = (pin_y - p[1] - g0) / P  # in pitches from the rack's reference gap
    off = abs(gap_local - round(gap_local)) * P
    return bool(off <= (GAP - PIN_D) / 2 + 0.0005)


def opening(objects, name) -> float:
    """How far the rack (its finger) is open from its stop (m)."""
    side = 1 if name == "far" else -1
    p, _ = rack_in_housing(objects, name)
    return float(side * (p[1] - side * CLOSED_C))


def _rack_points(objects, name):
    side = 1 if name == "far" else -1
    r = objects[name]
    Rr, p = _rotm(r.quat_xyzw), np.asarray(r.pos, float)
    bx = side * ((T_OUT - R) + BODY_W / 2)
    loc = [(bx, y, z) for y in (-RL / 2, -RL / 4, 0.0, RL / 4, RL / 2) for z in (0.0, RH)] + [(bx, 0.0, RH + BOSS[2])]
    loc.append((bx, side * (RL / 2 + PLATE[1] / 2), PLATE[2]))
    return [p + Rr @ np.array(v) for v in loc]


def _hands(objects):
    return [np.asarray(h["tcp"], float) for h in objects["housing"].hands.values()]


def _touched(objects, name) -> bool:
    c = objects[name].contacts
    return bool(c and any(v for v in c.values()))


def handled(objects, name) -> bool:
    """A gripper is on or within about 5 mm of this rack."""
    if _touched(objects, name):
        return True
    pts = _rack_points(objects, name)
    return any(min(float(np.linalg.norm(t - q)) for q in pts) < 0.015 + NEAR_R for t in _hands(objects))


def hands_clear(objects) -> bool:
    if any(_touched(objects, n) for n in ("near", "far", "housing")):
        return False
    h = objects["housing"]
    Rh, ph = _rotm(h.quat_xyzw), np.asarray(h.pos, float)
    for t in _hands(objects):
        lp = Rh.T @ (t - ph)
        gap = np.maximum(np.abs(lp[:2]) - np.array([HW_OUT / 2, HL / 2 + PLATE[1]]), 0.0)
        gap = np.r_[gap, max(lp[2] - (RB + RH + BOSS[2]), 0.0)]
        if np.linalg.norm(gap) < CLEAR + 0.015:
            return False
    return True


HOLD_STEPS = int(round(SUCCESS_HOLD_S / STEP_S))  # 80 control steps


def _update(objects):
    """Once per control step: coupled travel, rack speed, and how long the full-score state has held."""
    step = int(objects["housing"].step)
    if _H["last"] is not None and step < _H["last"]:  # a new episode without a reset call: clear the record too
        _reset_record()
    if _H["last"] == step:
        return
    _H["last"] = step
    sn, sf = opening(objects, "near"), opening(objects, "far")
    both = meshed(objects, "near") and meshed(objects, "far")
    one_free = not handled(objects, "far")  # driven from the white rack or the pinion, the orange rack untouched
    prev = _H["prev"]
    _H["speed"] = 0.0 if prev is None else max(abs(sn - prev[0]), abs(sf - prev[1])) / STEP_S
    if prev is not None and both and one_free:
        dn, df = sn - prev[0], sf - prev[1]
        if dn > 0 and df > 0:
            _H["coupled"] += min(dn, df)
    _H["prev"] = (sn, sf)
    d = (sn - sf) / 2
    _H["ok"] = bool(both and abs(d) <= SYM)
    _H["full_now"] = bool(_H["ok"] and sn >= OPEN and sf >= OPEN and _H["coupled"] >= COUPLED
                          and _H["speed"] < STILL and hands_clear(objects))
    _H["held"] = _H.get("held", 0) + 1 if _H["full_now"] else 0


def full_now(objects) -> bool:
    """The full-score state at this step (before the 4 s hold)."""
    _update(objects)
    return bool(_H["full_now"])


def score(objects):
    """0: either rack not in mesh with the pinion (seated in its channel: pitch line within 1.5 mm, underside within
    1.5 mm of its rail, within 3 degrees of level and 4 of straight; over the pinion: the pinion's axis between the
    rack's first and last tooth; in mesh: the pin at the mesh point in a tooth gap), or the fingers not even
    (|s_near - s_far| / 2 over 2 mm: a wrong tooth is 4.71 mm). 1.0: in mesh and even, both fingers at least 15 mm open,
    each rack having travelled at least 15 mm apart while the orange rack was untouched (so driven from the white rack
    or the pinion), the racks still and every gripper clear (30 mm), all of it held for the last 4 s (80 control steps,
    counted here, so a run that ends without the hold scores 0.5). 0.5: in mesh and even otherwise."""
    _update(objects)
    sn, sf = opening(objects, "near"), opening(objects, "far")
    d = (sn - sf) / 2
    if not _H["ok"]:
        s = 0.0
    elif _H["held"] >= HOLD_STEPS:
        s = 1.0
    else:
        s = 0.5
    st = int(objects["housing"].step)
    if _TRACE and st % _EVERY == 0 and _H.get("traced") != st:
        _H["traced"] = st
        pn, _ = rack_in_housing(objects, "near")
        pf, _ = rack_in_housing(objects, "far")
        with open(_TRACE, "a") as f:
            f.write(json.dumps({
                "step": st, "s_near_mm": round(sn * 1000, 2), "s_far_mm": round(sf * 1000, 2), "d_mm": round(d * 1000, 2),
                "near_rel_mm": [round(float(v) * 1000, 2) for v in pn], "far_rel_mm": [round(float(v) * 1000, 2) for v in pf],
                "seated": [seated(objects, "near"), seated(objects, "far")],
                "overlap": [overlaps(objects, "near"), overlaps(objects, "far")],
                "meshed": [meshed(objects, "near"), meshed(objects, "far")], "coupled_mm": round(_H["coupled"] * 1000, 2),
                "far_tilt_yaw_deg": [round(math.degrees(math.acos(float(np.clip(rack_in_housing(objects, "far")[1][2, 2], -1, 1)))), 2),
                                     round(math.degrees(math.atan2(rack_in_housing(objects, "far")[1][1, 0], rack_in_housing(objects, "far")[1][0, 0])), 2)],
                "pinion_deg": round(math.degrees(objects["housing"].q["pinion"]), 2),
                "housing": [round(float(v), 4) for v in objects["housing"].pos],
                "touch": {n: sorted({x for v in (objects[n].contacts or {}).values() for x in v}) for n in ("near", "far", "housing")},
                "handled": [handled(objects, "near"), handled(objects, "far")], "clear": hands_clear(objects),
                "full_now": _H["full_now"], "held_steps": _H["held"],
                "hands": [[round(float(v), 4) for v in t] for t in _hands(objects)], "score": s}) + "\n")
    return s


def success(objects):
    """yam_arena ends the episode once this has held for SUCCESS_HOLD_S (4 s); score() counts the same hold itself."""
    return full_now(objects)


def distance(objects):
    return 1.0 - score(objects)


# -- scripted demo (open loop: fixed waypoints, corrected for the arms' measured sag) ----------------------------------
def _rest(side, t, finger, label) -> Waypoint:
    """Parked out to the arm's own side, gripper straight down, well clear of the gripper being assembled."""
    return Waypoint(t, (0.20, 0.25 if side == "left" else -0.25, TOP + 0.16), finger, label)


TIP_UP = 0.004  # pinches: pad tips this far above the boss's foot
SAG_L = (_env("SAG_LX", 0.0), _env("SAG_LY", 0.0), _env("SAG_LZ", 0.0))  # left arm: commanded minus reached (m)
SAG_R = (_env("SAG_RX", 0.0), _env("SAG_RY", 0.0), _env("SAG_RZ", 0.0))
PUSH = _env("PUSH", 0.002)  # the seated rack is pressed this much further down before the fingers open
PULL = _env("PULL", 0.021)  # the white rack is pulled this far open (from one tooth to about 30 mm)


def _wp(t, p, finger, label, sag=(0, 0, 0)):
    return Waypoint(t, (float(p[0] + sag[0]), float(p[1] + sag[1]), float(p[2] + sag[2])), finger, label)


def demo():
    """Left hand: pinch the orange rack's boss on the bench, carry it over the far channel at the white rack's opening
    (one tooth: where the fingers sit evenly), lower it onto the pins and press it home, let go. Right hand, after:
    pinch the white rack's boss and pull it 21 mm further open (the orange rack follows through the pinion), let go."""
    o, c = FINGER_OPEN, FINGER_CLOSED
    g = -0.015  # pads 30 mm apart
    bx = (T_OUT - R) + BODY_W / 2
    # left: the orange rack's boss on the bench, then over its seat
    b0 = np.array([FAR0[0] + bx, FAR0[1], FAR0[2] + RH + TIP_UP + 0.013])
    seat = np.array([XH + R + bx, YH + CLOSED_C + START_OPEN, TOP + RB + RH + TIP_UP + 0.013])
    up = np.array([0, 0, 0.05])
    left = [_wp(3.0, b0 + up, g, "above the orange rack's boss", SAG_L),
            _wp(1.5, b0, g, "fingers either side of the boss", SAG_L),
            _wp(0.8, b0, c, "pinch it", SAG_L),
            _wp(1.2, b0 + up, c, "lift the rack", SAG_L),
            _wp(2.5, seat + np.array([0, 0, 0.025]), c, "over the far channel, at the white rack's opening", SAG_L),
            _wp(1.5, seat + np.array([0, 0, 0.010]), c, "lower: teeth just above the pins", SAG_L),
            _wp(1.5, seat + np.array([0, 0, 0.004]), c, "onto the pins", SAG_L),
            _wp(1.2, seat - np.array([0, 0, PUSH]), c, "press it home", SAG_L),
            _wp(0.8, seat - np.array([0, 0, PUSH]), g, "let go", SAG_L),
            _wp(1.0, seat + up, g, "up", SAG_L),
            _rest("left", 2.0, o, "back to the start")]
    t_left = sum(w.duration for w in left) + 0.5
    # right: the white rack's boss, then pull it further open (-y)
    nb = np.array([XH - R - bx, YH - CLOSED_C - START_OPEN, TOP + RB + RH + TIP_UP + 0.013])
    right = [_rest("right", t_left - 2.5, o, "wait"),
             _wp(2.5, nb + up, g, "above the white rack's boss", SAG_R),
             _wp(1.5, nb, g, "fingers either side of the boss", SAG_R),
             _wp(0.8, nb, c, "pinch it", SAG_R),
             _wp(2.5, nb - np.array([0, PULL, 0]), c, "pull the white finger open", SAG_R),
             _wp(0.8, nb - np.array([0, PULL, 0]), g, "let go", SAG_R),
             _wp(1.0, nb - np.array([0, PULL, 0]) + up, g, "up", SAG_R),
             _rest("right", 2.0, o, "back to the start")]
    T = max(sum(w.duration for w in left), sum(w.duration for w in right)) + SUCCESS_HOLD_S + 3.0
    for arm in (left, right):
        arm.append(Waypoint(T - sum(w.duration for w in arm), arm[-1].pos, arm[-1].finger, "wait", arm[-1].yaw))
    return {"left": left, "right": right}
