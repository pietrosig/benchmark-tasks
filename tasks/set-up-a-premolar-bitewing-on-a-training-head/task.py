"""A dental training head is clamped on the bench, reclined 25 degrees, its mouth open. A bitewing holder (a Rinn XCP-ORA arm and
aiming ring with a red bitewing bite block holding a size 2 phosphor plate) lies on a tray beside it. Set up a right
premolar bitewing: the plate upright between the tongue space and the lower right premolars, parallel to them, its
front edge at the canine, the bite tab on the lower premolars; then close the jaw onto the tab (on a training head the
operator closes the jaw; on a patient the patient bites while the operator holds the instrument still) and let go.

Run it:  inspect-robots run --embodiment yam_arena --policy <policy> -E task=task.py \\
           --instruction "$(cat instruction.txt)" --scorer task_score \\
           -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 --max-steps 1800
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
import head as H  # noqa: E402
from mouthkit import Slider, fixture, start_at  # noqa: E402  (importing mouthkit hooks yam_arena)
from yam_arena import ik  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"


def _env(name, default):
    return float(os.environ.get(name, default))


SEED = int(os.environ.get("TASK_SEED", "0"))

# -- the training head, clamped on the bench --------------------------------------------------------------------------
_PV = H.to_world(H.PIV)
MOUNT = [((_PV[0], 0.0, _PV[2] / 2), (0.028, 0.028, _PV[2]), H.METAL_C, {"shape": "cyl", "axis": "Z"}),  # pillar
         ((_PV[0], 0.0, _PV[2]), (0.036, 0.036, 0.036), H.METAL_C, {"shape": "cyl", "axis": "Y", "collide": False}),  # ball joint
         ((_PV[0] + 0.02, 0.0, 0.005), (0.090, 0.070, 0.010), H.METAL_C)]  # clamp plate on the bench
LAMP = [((0.17, 0.0, 0.34), (0.01, 0.01, 0.01), (1.0, 0.96, 0.88),  # the operatory light, in front of and above the face
         {"shape": "light", "radius": 0.04, "intensity": _env("LAMP", 15000.0)})]
JAW = Slider("jaw", tuple(H.HINGE), "Y", tuple(H.jaw_items()), low=-H.OPEN, high=0.0, kind="revolute", stiffness=0.0,
             damping=0.02, friction=H.JAW_FRICTION[0], dynamic_friction=H.JAW_FRICTION[1], mass=H.JAW_MASS)
HEAD = fixture("head", (0.0, 0.0, TOP), [*H.head_items(), *MOUNT, *LAMP], [JAW], friction=0.8)
start_at("head", "jaw", -H.OPEN)

# -- the holder on its tray -------------------------------------------------------------------------------------------
C_T, R_T = H.target_frame()  # target, in the jaw's own (closed) frame
U_T, V_T, N_T = R_T[:, 0], R_T[:, 1], R_T[:, 2]
_RH = H.RREC.T @ R_T  # the target frame before the head's recline (u and n level)
R_START = np.c_[_RH[:, 0], -_RH[:, 2], np.array([0.0, 0.0, 1.0])]  # lying on its plate face, the ring above it
TRAY_C = np.array([0.255 + 0.01 * (SEED % 3 - 1), 0.262, 0.008])
TRAY = Box("tray", pos=(float(TRAY_C[0]), float(TRAY_C[1]), TOP + 0.004), size=(0.20, 0.13, 0.008), color=(0.93, 0.93, 0.95))
# it rests tipped onto the plate's edge and the bite block's slot (measured in the simulator after settling)
R_START = H._rz(math.radians(-8.0)) @ R_START
R_START = H._quat_to_R((0.0287, -0.1738, -0.0026, 0.9844)) @ R_START
P_START = np.array([TRAY_C[0] + 0.018, TRAY_C[1] + 0.0047, 0.0174])
HOLDER = fixture("holder", (P_START[0], P_START[1], TOP + P_START[2]), H.holder_items_and_start(TRAY_C, R_START), [],
                 fixed=False, base_mass=H.HOLDER_MASS, friction=1.2, com=R_START @ (H.HOLDER_COM / 1000.0))
OBJECTS = [HEAD, TRAY, HOLDER]

# -- scoring ----------------------------------------------------------------------------------------------------------
TOL_FULL = dict(u=0.004, v=0.003, d=(0.002, 0.012), h=10.0, t=10.0, r=10.0)
TOL_LOOSE = dict(u=0.008, v=0.006, d=(0.000, 0.016), h=20.0, t=20.0, r=20.0)
BITTEN_Q = -0.065  # jaw angle (rad) above which the jaw counts as closed on the tab: incisal gap 6 mm or less
CLEAR_M = 0.030  # let go: both tool points at least this far from every part of the holder
LIP_X = H.XI - 0.014  # the mask's lip line (front of the lips)
_TRACE = os.environ.get("TASK_TRACE")
_st = {"calls": 0, "lip": False, "still_n": 0, "prev": None}


def _quat_to_R(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def holder_pose(objects):
    """The plate's centre (bench coordinates) and the holder frame [u v n]."""
    h = objects["holder"]
    p = np.asarray(h.pos, float) - np.array([0.0, 0.0, TOP])
    return p, _quat_to_R(h.quat_xyzw) @ R_START


def jaw_q(objects) -> float:
    return float(objects["head"].q["jaw"])


def _angle(a, b) -> float:
    """Angle between two vectors, degrees (0..180)."""
    c = float(a @ b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def errors(objects) -> dict:
    """The plate's pose in the lower jaw's frame against the target. Position: along the row (u), vertical (v), and its
    distance lingual of the premolars' lingual faces (d). Orientation, in degrees, one angle per axis:
    h, horizontal angulation (the face normal turned about the vertical; 180 if it faces away from the teeth);
    t, tilt of the face from upright (about the row); r, rotation within the plate's own plane (about its normal): the
    plate's long edge against the row, 0 when the bite tab is at the front, on the teeth' side, 90 or 180 when the plate
    is turned so the tab points up, down or back."""
    p, R = holder_pose(objects)
    Rj = H.jaw_rot(jaw_q(objects))
    pj = H.HINGE + Rj.T @ (p - H.HINGE)
    Rjh = Rj.T @ R
    e = pj - C_T
    u, n = Rjh[:, 0], Rjh[:, 2]
    nh = n - V_T * (n @ V_T)
    h = _angle(nh, N_T) if np.linalg.norm(nh) > 1e-9 else 90.0
    t = math.degrees(math.asin(min(1.0, abs(float(n @ V_T)))))
    up = U_T - n * float(U_T @ n)  # the row's direction laid into the plate's plane
    r = _angle(u, up) if np.linalg.norm(up) > 1e-9 else 90.0
    return {"u": float(e @ U_T), "v": float(e @ V_T), "d": H.D_TARGET - float(e @ N_T), "h": h, "t": t, "r": r}


RING_OUT = H.RING[2] / 1000.0 + (H.RING[3] - H.RING[2]) / 2000.0  # 75 mm: the ring's centre out along the plate normal


TO_HEAD = np.array([[0.906, 0.406], [-0.403, 0.914]])  # bench (plan, tilt change) -> head's (h, t); see ring_reading


def _plan(v) -> float:
    return math.atan2(float(v[1]), float(v[0]))


def ring_reading(objects) -> dict:
    """The real-trial worksheet, computed from the RING and from angles taken the way a person takes them (README
    "After each trial").

    A person measures two bench angles of the ring face against its calibration marks: its plan angle against the
    base line (protractor; + when the ring has turned to face more toward the robot, i.e. its front edge swung toward
    the face) and its tilt from plumb minus the tilt recorded at calibration (level app; + when its top leans further
    toward the face). The head is reclined, so these are converted to the head's axes:
    h = 0.906 x plan + 0.406 x tilt change, t = -0.403 x plan + 0.914 x tilt change (worksheet: 0.9 and 0.4).
    The ring centre's offset from its calibrated place (along the row and along the tick line, read on the photo, and
    out from the cheek along the beam axis) is then corrected for the turn, because the ring sits 75 mm out along the
    plate normal: u = offset along the row - 75 mm x sin h (about 1.3 mm per degree), v = offset along the tick line -
    75 mm x sin t, d = 7 mm - outward shift - 0.0114 mm x (h^2 + t^2) [degrees]. The calibration reference is the
    target's normal at the current jaw angle."""
    p, R = holder_pose(objects)
    Rj = H.jaw_rot(jaw_q(objects))
    ring = H.HINGE + Rj.T @ (p + R[:, 2] * RING_OUT - H.HINGE)  # the ring's centre in the jaw's frame
    off = ring - (C_T + N_T * RING_OUT)
    n_w, ref_w = R[:, 2], Rj @ N_T  # bench coordinates: the ring face's normal and its calibrated direction
    plan = math.degrees(_plan(n_w) - _plan(ref_w))
    plan = (plan + 180.0) % 360.0 - 180.0
    tilt = math.degrees(math.asin(max(-1.0, min(1.0, float(n_w[2])))) - math.asin(max(-1.0, min(1.0, float(ref_w[2])))))
    h, t = (float(x) for x in TO_HEAD @ np.array([plan, tilt]))
    u = float(off @ U_T) - RING_OUT * math.sin(math.radians(h))
    v = float(off @ V_T) - RING_OUT * math.sin(math.radians(t))
    d = H.D_TARGET - float(off @ N_T) - 0.0114e-3 * (h * h + t * t)
    return {"u": u, "v": v, "d": d, "h": h, "t": t, "plan": plan, "tilt": tilt}


def within(e, tol) -> bool:
    return (abs(e["u"]) <= tol["u"] and abs(e["v"]) <= tol["v"] and tol["d"][0] <= e["d"] <= tol["d"][1]
            and e["h"] <= tol["h"] and e["t"] <= tol["t"] and e["r"] <= tol["r"])


def on_premolars(e) -> bool:
    """The tab rests on the lower right premolars or first molar: centred on the row within 12 mm along it, at tab
    height, lingual of the teeth, and oriented within the loose band (h, t and r each 20 degrees or less), so the tab is
    on the teeth' side, not turned up, down or back."""
    return (-0.012 <= e["u"] <= 0.012 and abs(e["v"]) <= 0.006 and 0.0 <= e["d"] <= 0.020
            and e["h"] <= 20.0 and e["t"] <= 20.0 and e["r"] <= 20.0)


def _box_dist(p, lo, hi):
    return float(np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0)))


def hand_gap(objects, hand) -> float:
    """A gripper's tool point to the nearest part of the holder (m)."""
    p, R = holder_pose(objects)
    q = R.T @ (np.asarray(hand["tcp"], float) - np.array([0.0, 0.0, TOP]) - p) * 1000.0
    d = min(_box_dist(q, np.array([b[1][0], b[2][0], b[3][0]]), np.array([b[1][1], b[2][1], b[3][1]]))
            for b in H.HOLDER_BOXES)
    r0, r1, n0, n1 = H.RING
    rr = math.hypot(q[0], q[1])
    dr = max(r0 - rr, rr - r1, 0.0)
    dn = max(n0 - q[2], q[2] - n1, 0.0)
    return min(d, math.hypot(dr, dn)) / 1000.0


def released(objects) -> bool:
    return all(hand_gap(objects, h) >= CLEAR_M for h in objects["head"].hands.values())


def bitten(objects) -> bool:
    return jaw_q(objects) >= BITTEN_Q


def in_mouth(hand) -> bool:
    """A gripper's fingertips (the pad tips, 13 mm past its tool point) past the front of the lips, inside the mouth
    opening: a finger in the patient's mouth."""
    tip = np.asarray(hand.get("tips", hand["tcp"]), float)
    x, y, z = H.from_world(tip - np.array([0.0, 0.0, TOP]))
    return x > LIP_X and x < H.XI + 0.060 and abs(y) < H.MOUTH_W and H.ZO - 0.045 < z < H.ZO + 0.013


STILL_M = 0.001  # still: per 50 ms step, the plate's centre and the arm's end each move under 1 mm ...
STILL_RAD = math.radians(0.5)  # ... and the holder turns under 0.5 degrees; for 20 steps (1 s) running


def _marks(objects) -> np.ndarray:
    p, R = holder_pose(objects)
    return np.array([p, p + R @ (H.ROD_END / 1000.0)])


def _update(objects) -> None:
    if any(in_mouth(h) for h in objects["head"].hands.values()):
        _st["lip"] = True
    m = _marks(objects)
    R = holder_pose(objects)[1]
    prev, prev_r = _st.get("prev"), _st.get("prev_r")
    turned = 0.0 if prev_r is None else math.acos(max(-1.0, min(1.0, (float(np.trace(prev_r.T @ R)) - 1.0) / 2.0)))
    moved = prev is None or float(np.linalg.norm(m - prev, axis=1).max()) >= STILL_M or turned >= STILL_RAD
    _st["prev"], _st["prev_r"] = m, R
    _st["still_n"] = 0 if moved else _st["still_n"] + 1


def score(objects):
    return _score(objects)


def _score(objects):
    """0 for the rest of the run once a gripper's fingertips pass the front of the lips into the mouth. Otherwise, at the
    end: 1.0 plate within the full tolerance, jaw closed on the tab, both grippers 3 cm clear, all still for 1 s;
    0.75 the same within the loose tolerance; 0.5 within the full tolerance but the jaw open or a gripper on the
    holder; 0.25 the tab resting on the lower right premolars, the holder oriented within 20 degrees on all three axes
    (so a plate turned within its plane, facing away or upside down gets nothing); 0 otherwise. Tolerances: TOL_FULL,
    TOL_LOOSE (u, v, d in m; h, t, r in degrees, see errors())."""
    _update(objects)
    _trace(objects)
    if _st["lip"]:
        return 0.0
    e = errors(objects)
    settled = released(objects) and _st["still_n"] >= 20
    if bitten(objects) and settled and within(e, TOL_FULL):
        return 1.0
    if bitten(objects) and settled and within(e, TOL_LOOSE):
        return 0.75
    if within(e, TOL_FULL):
        return 0.5
    if on_premolars(e):
        return 0.25
    return 0.0


def distance(objects):
    e = errors(objects)
    return float(math.sqrt(e["u"] ** 2 + e["v"] ** 2 + (e["d"] - H.D_TARGET) ** 2))


def _trace(objects) -> None:
    _st["calls"] += 1
    if not _TRACE or _st["calls"] % 5:
        return
    e = errors(objects)
    hands = objects["head"].hands
    with open(_TRACE, "a") as f:
        f.write(json.dumps({
            "call": _st["calls"], "jaw_deg": round(math.degrees(jaw_q(objects)), 2),
            "err_mm": {k: round(v * 1000, 1) for k, v in e.items() if k in "uvd"},
            "ang_deg": {"h": round(e["h"], 1), "t": round(e["t"], 1), "r": round(e["r"], 1)},
            "bitten": bitten(objects), "released": released(objects), "still_n": _st["still_n"], "lip": _st["lip"],
            "plate_mm": [round(float(v) * 1000, 1) for v in holder_pose(objects)[0]],
            "hq": [round(float(v), 4) for v in objects["holder"].quat_xyzw],
            "hands": {k: {"tcp": [round(float(v), 3) for v in np.asarray(h["tcp"]) - np.array([0, 0, TOP])],
                          "open": round(h["open"], 2), "gap_mm": round(hand_gap(objects, h) * 1000, 1)}
                      for k, h in hands.items()},
        }) + "\n")


# -- scripted demo ----------------------------------------------------------------------------------------------------
_down_grip = _demo.grip_rotation


def _grip_rotation(yaw=0.0):
    return np.array(yaw, float).reshape(3, 3) if np.ndim(yaw) else _down_grip(yaw)


_demo.grip_rotation = _grip_rotation


def _tool(approach, closing) -> np.ndarray:
    z = np.array(approach, float) / np.linalg.norm(approach)
    x = np.array(closing, float)
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    return np.c_[x, np.cross(z, x), z]


def _wp(t, p, finger, label, R) -> Waypoint:
    return Waypoint(t, (float(p[0]), float(p[1]), float(p[2]) + TOP), finger, label, tuple(np.asarray(R).reshape(-1)))


def _rest(side, t, finger, label) -> Waypoint:
    p, r = ik.tcp_pose(np.array(READY, float), (0.0, 0.305 if side == "left" else -0.305, 0.75))
    return Waypoint(t, tuple(float(v) for v in p), finger, label, tuple(r.reshape(-1)))


# the left hand pinches the arm's straight leg outside the ring, fingers closing across it (along u)
HOLD_TILT = math.radians(30.0)  # the hand's approach: along the arm's leg toward the face, tilted 30 deg down off it
G_HOLD = np.c_[np.array([1.0, 0.0, 0.0]), np.zeros(3), -np.array([0.0, math.sin(HOLD_TILT), math.cos(HOLD_TILT)])]
G_HOLD[:, 1] = np.cross(G_HOLD[:, 2], G_HOLD[:, 0])  # tool frame in the holder frame: closing (x), y, approach (z)
SQUEEZE = _env("SQUEEZE", 0.005)  # fingers commanded this far past the 7 mm bar


def holder_tool(c, R):
    """The left hand's tool pose holding the holder (plate centre c, frame R) by the middle of its indicator rod."""
    return c + R @ (H.GRASP / 1000.0), R @ G_HOLD


def placed_pose(q, lift=0.0, fwd=0.0, out=0.0):
    """The holder's pose with the plate at the target in the jaw opened to angle q, moved lift (up, v), fwd (toward
    the front of the mouth, u) and out (n) in the jaw's frame. Bench coordinates."""
    Rj = H.jaw_rot(q)
    c = C_T + U_T * fwd + V_T * lift + N_T * out
    return H.HINGE + Rj @ (c - H.HINGE), Rj @ R_T


UNDER = np.array([H.XI + 0.010, 0.0, H.ZO - 0.062 - 0.010])  # the right hand's tool point under the chin (jaw closed)
G_CHIN = _tool((0.35, 0.90, -0.25), (1.0, 0.0, 0.0))  # from the patient's left, fingers closed, their top under the chin
LIFT = _env("LIFT", 0.018)  # the lifting hand is commanded this far above contact
OVERCLOSE = _env("OVERCLOSE", 0.05)  # the lifting hand aims this far (rad) past the tab stop, to press the jaw shut on it
PRESS = _env("PRESS", 0.001)  # the holding hand leans the tab this far into the lower teeth while the jaw closes
OUT = _env("OUT", 0.003)  # the demo sets the plate this far nearer the teeth than the target's middle (d 4 mm, in band)
FWD = _env("FWD", 0.003)  # ... and this far toward the front, against the drift back along the row as the jaw closes


def chin_tool(q, lift=0.0):
    return H.jaw_to_bench(q, UNDER) + np.array([0.0, 0.0, lift])


def demo():
    o = FINGER_OPEN
    shut = FINGER_CLOSED
    hold = FINGER_OPEN * ((0.007 - SQUEEZE) / 0.094)  # fingers closed on the 7 mm bar
    q0 = -H.OPEN
    qb = _env("QB", -0.040)  # where the jaw stops on the tab
    # left hand: pick the holder up by arm_c
    p0, r0 = holder_tool(P_START, R_START)
    left = [_wp(3.0, p0 - 0.035 * r0[:, 2], o, "above the leg of the holder's arm", r0),
            _wp(1.5, p0, o, "fingers round the arm", r0),
            _wp(0.8, p0, hold, "pinch it", r0),
            _wp(1.5, p0 + np.array([0, 0, 0.03]), hold, "lift", r0)]
    # carry it to the front of the mouth, turned upright and tilted with the open jaw, plate raised and forward
    steps = [(3.0, 0.021, 0.075, 0.0, "in front of the mouth, plate upright"),
             (2.5, 0.021, 0.040, 0.0, "in past the lips, between the front teeth"),
             (2.0, 0.021, 0.000, 0.0, "back over the premolars"),
             (2.0, 0.0005, FWD, OUT, "down: tab onto the lower premolars")]
    for t, lift, fwd, out, label in steps:
        c, R = placed_pose(q0, lift, fwd, out)
        p, r = holder_tool(c, R)
        left.append(_wp(t, p, hold, label, r))
    t_placed = sum(w.duration for w in left)
    # right hand, fingers closed: once the holder is in, from the patient's left, under the chin
    pc = chin_tool(q0, -0.006)
    right = [_rest("right", t_placed - 2.0, o, "wait while the holder goes in"),
             _wp(3.5, pc + np.array([0.0, -0.07, -0.004]), shut, "beside the chin, below it", G_CHIN),
             _wp(2.0, pc, shut, "fingers under the chin", G_CHIN),
             _wp(1.0, chin_tool(q0, 0.0), shut, "up to the chin", G_CHIN)]
    left.append(_wp(sum(w.duration for w in right) - t_placed + 0.3, np.array(left[-1].pos) - np.array([0, 0, TOP]),
                    hold, "hold the plate in place", np.array(left[-1].yaw).reshape(3, 3)))
    right.append(_wp(0.3, pc, shut, "ready", G_CHIN))
    # close together: the chin lifted along its arc and the holder carried with the lower jaw on the same schedule,
    # its tab leaned PRESS onto the lower premolars (so the hand follows the teeth instead of lifting the holder)
    n = 90
    for i in range(1, n + 1):
        q = q0 + (qb - q0) * (0.5 - 0.5 * math.cos(math.pi * i / n))
        right.append(_wp(0.1, chin_tool(q, LIFT), shut, "lift the chin: close the jaw onto the tab", G_CHIN))
        c, R = placed_pose(q, -PRESS, FWD, OUT)
        p, r = holder_tool(c, R)
        left.append(_wp(0.1, p, hold, "keep the plate where it is as the jaw closes", r))
    c, R = placed_pose(qb, -PRESS, FWD, OUT)
    p, r = holder_tool(c, R)
    pq, pe, pz = chin_tool(qb, LIFT), chin_tool(qb + OVERCLOSE, LIFT), chin_tool(qb, 0.0)
    # the right hand presses the jaw shut on the tab and eases off (the hinge's friction keeps it shut); only then does
    # the left hand let go of the holder, now held by the bite
    right += [_wp(1.0, pe, shut, "press the jaw shut on the tab", G_CHIN), _wp(1.0, pe, shut, "hold", G_CHIN),
              _wp(1.5, pz, shut, "ease off: the hinge's friction keeps the jaw shut", G_CHIN),
              _wp(1.0, pz - np.array([0, 0, 0.03]), shut, "down", G_CHIN),
              _wp(1.5, pz + np.array([0.0, -0.08, -0.03]), shut, "out", G_CHIN), _rest("right", 3.0, o, "rest")]
    c0, R0 = placed_pose(qb, 0.0005, FWD, OUT)
    p0_, r0_ = holder_tool(c0, R0)
    left += [_wp(2.0, p, hold, "hold the holder while the jaw is pressed shut", r),
             _wp(2.5, p0_, hold, "stop leaning on the teeth", r0_), _wp(2.0, p0_, o, "let go", r0_),
             _wp(1.5, p - 0.05 * r[:, 2], o, "back off the arm", r),
             _wp(2.0, p - 0.05 * r[:, 2] + np.array([0.0, 0.0, 0.06]), o, "up", r), _rest("left", 3.0, o, "rest")]
    T = max(sum(w.duration for w in left), sum(w.duration for w in right)) + 2.0
    for arm in (left, right):
        arm.append(Waypoint(T - sum(w.duration for w in arm), arm[-1].pos, arm[-1].finger, "wait", arm[-1].yaw))
    return {"left": left, "right": right}
