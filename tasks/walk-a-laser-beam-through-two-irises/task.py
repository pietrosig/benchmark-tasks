"""A free-space laser beam on an optical bench has drifted off its two alignment irises, and the robot walks it back.

On an aluminium plate, 20 cm above the bench top: a collimated Class 2 (under 1 mW) 650 nm dot laser module fires toward the
robot onto mirror M1 (left), which folds the beam across to mirror M2 (right), which sends it away from the robot through
two irises closed to their 0.8 mm minimum, 12 and 24 cm beyond M2, onto a photodiode whose photocurrent a multimeter
reads. Each mirror sits in a kinematic mount (Thorlabs
KM100) with two 1/4-80 adjuster knobs on its back, toward the robot: the upper knob tips the mirror up and down, the
lower one turns it left and right. Every knob has been turned off its aligned setting (a seeded draw), so the beam misses
the first iris and the meter reads about zero.

The ability tested is beam walking: two mirrors, four coupled adjusters, two irises. M1 mainly moves the spot on the
near iris and M2 the angle onto the far one, but each knob moves both, so the robot must alternate. Each knob is
gripped from behind and turned in fractions of a turn (one turn tilts the mirror 8 mrad: the beam moves about 4 mm per
turn at the far iris), with the mounts left undisturbed.

The beam is not simulated as light: its path is ray-traced every step from the four knob angles (exact reflection off
each mirror's plane), and the photodiode's reading is the Gaussian beam's transmission through both irises, relative
to the aligned reference. The knobs are real revolute joints with friction that the grippers turn.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --instruction "$(cat instruction.txt)" \\
           --scorer task_score -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 --max-steps 3000
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from beamkit import Slider, compute_targets, fixture, start_at  # noqa: E402  (importing beamkit hooks yam_arena)
from yam_arena import demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 5.0  # success must hold 5 s (hands clear, meter full), so the end of the video shows it
SCORE_OVER = "end"


def _env(name, default):
    return type(default)(os.environ.get(name, default))


# -- colours --------------------------------------------------------------------------------------------------------------
BLACK, DARK, ALU, STEEL, GOLD, RED, MIRROR, YELLOW, GREEN, WHITE = (
    (0.05, 0.05, 0.06), (0.14, 0.14, 0.16), (0.55, 0.57, 0.60), (0.70, 0.72, 0.75), (0.80, 0.62, 0.25),
    (1.0, 0.05, 0.03), (0.86, 0.88, 0.92), (0.95, 0.78, 0.10), (0.15, 0.85, 0.20), (0.93, 0.93, 0.91))
VB = {"collide": False}
CYL_X = {"shape": "cyl", "axis": "X"}
CYL_Z = {"shape": "cyl", "axis": "Z"}

# -- the optical layout (fixture frame = world x, y; z above the bench top) -------------------------------------------------
HB = 0.20  # beam height
LASER = np.array([0.56, 0.11, HB])  # the laser's exit aperture; it fires along -x
M1 = np.array([0.28, 0.11, HB])  # mirror centres (front surface)
M2 = np.array([0.28, -0.11, HB])
A1 = np.array([0.40, -0.11, HB])  # iris centres (their faces toward M2)
A2 = np.array([0.52, -0.11, HB])
DET = np.array([0.555, -0.11, HB])
YAW1, YAW2 = -math.pi / 4, math.pi / 4  # each mount's yaw: its local +x is the mirror's normal (front)

# -- the kinematic mount (Thorlabs KM100, in its local frame: x forward out of the mirror, origin at the mirror's centre) --
MRAD_PER_TURN = 8.0e-3  # mirror tilt per knob turn (1/4-80 adjuster, 0.3175 mm per turn, on a ~40 mm lever)
KNOB_D, KNOB_L = 0.012, 0.010  # knurled knob
KNOB_X = -0.032  # knob centre: 27-37 mm behind the mirror's face
KNOB_OFF = 0.0127  # adjusters at the plate's corners, 12.7 mm from the centre each way
KNOB_FRICTION = _env("KNOB_FRICTION", 0.03)  # N m: the adjuster's thread and spring preload
KNOBS = {"up": (KNOB_OFF, KNOB_OFF), "side": (-KNOB_OFF, -KNOB_OFF)}  # (local y, local z) of the upper and lower knob
MOUNT_BOXES = (
    ((-0.003, 0, 0), (0.006, 0.0254, 0.0254), MIRROR, CYL_X),  # the Ø1 in mirror
    ((-0.010, 0, 0), (0.010, 0.038, 0.038), BLACK),  # front plate
    ((-0.020, 0, 0), (0.008, 0.038, 0.038), BLACK),  # back plate
    ((-0.026, KNOB_OFF, KNOB_OFF), (0.004, 0.0064, 0.0064), STEEL, CYL_X),  # adjuster shanks
    ((-0.026, -KNOB_OFF, -KNOB_OFF), (0.004, 0.0064, 0.0064), STEEL, CYL_X),
)
POST_TOP = -0.019  # the post meets the mount's underside


def _mount_part(name, centre, yaw):
    """The mount on its post, as a part turned ``yaw`` about the vertical (held in place by a stiff, fixed joint)."""
    post_len = HB + POST_TOP
    boxes = MOUNT_BOXES + (
        ((-0.010, 0, POST_TOP - (post_len - 0.0127) / 2), (0.0254, 0.0254, post_len - 0.0127), STEEL, CYL_Z),  # Ø1 in pedestal post
        ((-0.010, 0, -HB + 0.0127 + 0.004), (0.020, 0.050, 0.008), BLACK),  # clamping fork, bolted to the board
    )
    return Slider(name, tuple(centre), "Z", boxes, -1e-5, 1e-5, kind="revolute", yaw=yaw, stiffness=1e4, damping=50.0,
                  mass=0.3, gravity=False)


def _knob_part(mount, key):
    y, z = KNOBS[key]
    boxes = (((0, 0, 0), (KNOB_L, KNOB_D, KNOB_D), DARK, CYL_X),
             ((0.0, 0, KNOB_D / 2 - 0.001), (KNOB_L, 0.002, 0.002), WHITE, VB))  # an index line, to see it turn
    return Slider(f"{mount}_{key}", (KNOB_X, y, z), "X", boxes, -40.0, 40.0, kind="revolute", parent=mount,
                  stiffness=0.0, damping=0.002, friction=KNOB_FRICTION, mass=0.01, gravity=False)


# -- spots and meter: display parts, their targets computed from the beam each step (no collision) ----------------------
SPOT = ((0, 0, 0), (0.001, 0.006, 0.006), RED, {**CYL_X, "collide": False})  # drawn larger than the beam, to be seen


def _spot_parts(tag, base):
    y = Slider(f"{tag}_y", tuple(base), "Y", (), -0.025, 0.025, stiffness=2000.0, damping=40.0, mass=0.002,
               gravity=False, computed=True)
    z = Slider(f"{tag}_z", (0, 0, 0), "Z", (SPOT,), -0.025, 0.025, parent=f"{tag}_y", stiffness=2000.0, damping=40.0,
               mass=0.002, gravity=False, computed=True)
    return [y, z]


METER = np.array([0.45, -0.25, 0.0])  # a multimeter lying on the bench beside the beam
BAR_L = 0.070
BAR = Slider("bar", (METER[0] + 0.005, METER[1] - 0.075, 0.0315), "Y",
             (((0, BAR_L / 2, 0), (0.022, BAR_L, 0.002), GREEN, VB),), 0.0, BAR_L, stiffness=2000.0, damping=40.0,
             mass=0.002, gravity=False, computed=True)

def _pedestal(x, y, top):
    """A Ø1 in pedestal post from the plate up to ``top``, its clamping fork bolted to the plate."""
    return [((x, y, (0.0127 + top) / 2), (0.0254, 0.0254, top - 0.0127), STEEL, CYL_Z),
            ((x, y, 0.0127 + 0.004), (0.050, 0.020, 0.008), BLACK)]


IRIS_OD = 0.0308  # Edmund #53-914 mounted iris: 30.8 mm outside diameter, 0.8-12 mm aperture


def _iris(c):
    return [((c[0] + 0.004, c[1], c[2]), (0.008, IRIS_OD, IRIS_OD), BLACK, CYL_X),
            ((c[0] - 0.0002, c[1], c[2]), (0.0005, 0.016, 0.016), (0.30, 0.30, 0.32), {**CYL_X, "collide": False}),
            ((c[0] - 0.0004, c[1], c[2]), (0.0005, 0.0012, 0.0012), BLACK, {**CYL_X, "collide": False}),
            ((c[0] + 0.004, c[1] + 0.012, c[2] - 0.018), (0.003, 0.003, 0.012), STEEL),  # the lever, at its stop
            *_pedestal(c[0] + 0.004, c[1], c[2] - IRIS_OD / 2)]


BENCH_BOXES = [
    ((0.40, 0.0, 0.00635), (0.40, 0.46, 0.0127), DARK),  # aluminium plate, 12.7 mm thick, tapped at the post positions
    ((LASER[0] + 0.020, LASER[1], HB), (0.040, 0.012, 0.012), ALU, CYL_X),  # Ø12 x 40 mm laser module
    ((LASER[0] + 0.012, LASER[1], HB), (0.010, 0.030, 0.030), BLACK),  # its Ø1/2 in lens mount
    *_pedestal(LASER[0] + 0.012, LASER[1], HB - 0.015),
    ((LASER[0] - 0.0005, LASER[1], HB), (0.001, 0.004, 0.004), RED, {**CYL_X, "collide": False}),  # the exit spot
    # the two mounted irises, closed to their 0.8 mm stop (lever down against the stop), on pedestal posts
    *_iris(A1), *_iris(A2),
    # the photodiode in its black holder, on a pedestal post
    ((DET[0] + 0.008, DET[1], HB), (0.016, 0.025, 0.025), BLACK),
    *_pedestal(DET[0] + 0.008, DET[1], HB - 0.0125),
    # the multimeter (yellow), lying face up; its reading drawn as a green bar sliding out from under a cover
    ((METER[0], METER[1], 0.015), (0.090, 0.170, 0.030), YELLOW),
    ((METER[0] + 0.005, METER[1] + 0.030, 0.0303), (0.030, BAR_L + 0.004, 0.001), BLACK, VB),  # the reading's window
    ((METER[0] + 0.005, METER[1] - 0.040, 0.0335), (0.040, BAR_L, 0.003), YELLOW, VB),  # cover: hides the bar at 0
    # the beam where it doesn't move visibly: laser to M1, M1 to M2 (2 mm red rods, drawn only)
    (((LASER[0] + M1[0]) / 2, LASER[1], HB), (LASER[0] - M1[0], 0.002, 0.002), RED, VB),
    ((M1[0], 0.0, HB), (0.002, M1[1] - M2[1], 0.002), RED, VB),
]

_mounts = [_mount_part("m1", M1, YAW1), _mount_part("m2", M2, YAW2)]
_knobs = [_knob_part(m, k) for m in ("m1", "m2") for k in ("up", "side")]
_spots = _spot_parts("s1", A1 + np.array([-0.001, 0, 0])) + _spot_parts("s2", A2 + np.array([-0.001, 0, 0]))
BENCH = fixture("bench", (0.0, 0.0, TOP), BENCH_BOXES, [*_mounts, *_knobs, *_spots, BAR])
OBJECTS = [BENCH]
KNOB_NAMES = [k.name for k in _knobs]

# -- the starting misalignment: each knob turned off its aligned zero by a seeded number of turns ---------------------------
# A table of 20 draws (turns, for m1_up, m1_side, m2_up, m2_side), each 0.30-0.60 turn either way, chosen so the beam
# starts off the first iris. TASK_SEED picks one (default 0); the real trial uses the same table.
_RNG = np.random.default_rng(20261122)
_DRAWS = [tuple(float(s * m) for s, m in zip(_RNG.choice([-1, 1], 4), _RNG.uniform(0.30, 0.60, 4))) for _ in range(60)]
OFFSETS: list = []  # filled below with the first 20 draws that start the reading under 1%
SEED = _env("TASK_SEED", 0) % 20


# -- the beam ------------------------------------------------------------------------------------------------------------
W_EFF = 1.5e-3  # m: the module's beam 1/e^2 radius (about 3 mm across); through a 0.8 mm iris the transmitted power
HOLE_R = 0.4e-3  # falls off with the offset r as exp(-2 r^2 / w^2)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def mirror_normal(yaw, up_rad, side_rad) -> np.ndarray:
    """The mirror's normal (world) with its knobs at these angles (rad): the upper knob tips it about its horizontal
    axis, the lower one turns it about its vertical axis, MRAD_PER_TURN per turn."""
    tilt_v = MRAD_PER_TURN * up_rad / (2 * math.pi)
    tilt_h = MRAD_PER_TURN * side_rad / (2 * math.pi)
    return _rz(yaw) @ _rz(tilt_h) @ _ry(-tilt_v) @ np.array([1.0, 0, 0])


def _nominal_normal(c_in, c, c_out) -> np.ndarray:
    a = (c_in - c) / np.linalg.norm(c_in - c)
    b = (c_out - c) / np.linalg.norm(c_out - c)
    n = a + b
    return n / np.linalg.norm(n)


# the aligned mounts' exact normals (so that with every knob at zero the beam goes through both iris centres)
N1_0 = _nominal_normal(LASER, M1, M2)
N2_0 = _nominal_normal(M1, M2, A1)
_FIX1 = _rz(math.atan2(N1_0[1], N1_0[0]) - YAW1)  # tiny corrections from the 45 deg the mounts are drawn at
_FIX2 = _rz(math.atan2(N2_0[1], N2_0[0]) - YAW2)


def _hit(p, d, c, n):
    t = float(np.dot(c - p, n) / np.dot(d, n))
    return p + t * d


def _reflect(d, n):
    return d - 2 * np.dot(d, n) * n


def beam(q: dict) -> dict:
    """Ray-trace the beam from the laser off both mirrors to the two iris planes. ``q``: knob angles (rad)."""
    n1 = _FIX1 @ mirror_normal(YAW1, q["m1_up"], q["m1_side"])
    n2 = _FIX2 @ mirror_normal(YAW2, q["m2_up"], q["m2_side"])
    p, d = LASER.copy(), np.array([-1.0, 0, 0])
    h1 = _hit(p, d, M1, n1)
    d = _reflect(d, n1)
    h2 = _hit(h1, d, M2, n2)
    d = _reflect(d, n2)
    ex = np.array([1.0, 0, 0])
    a1 = _hit(h2, d, A1, ex)
    a2 = _hit(h2, d, A2, ex)
    o1, o2 = a1 - A1, a2 - A2
    on_m2 = float(np.linalg.norm(h2 - M2)) < 0.0115  # the beam lands on M2's Ø1 in face at all
    r1, r2 = float(np.hypot(o1[1], o1[2])), float(np.hypot(o2[1], o2[2]))
    power = math.exp(-2 * (r1 ** 2 + r2 ** 2) / W_EFF ** 2) if on_m2 and d[0] > 0 else 0.0
    return {"o1": o1, "o2": o2, "r1": r1, "r2": r2, "power": power, "on_m2": on_m2}


def _targets(fixture_name, part, q):
    if fixture_name != "bench":
        return 0.0
    b = beam(q)
    clip = 0.024
    return {"s1_y": float(np.clip(b["o1"][1], -clip, clip)), "s1_z": float(np.clip(b["o1"][2], -clip, clip)),
            "s2_y": float(np.clip(b["o2"][1], -clip, clip)), "s2_z": float(np.clip(b["o2"][2], -clip, clip)),
            "bar": BAR_L * b["power"]}.get(part, 0.0)


compute_targets(_targets)

OFFSETS[:] = [d for d in _DRAWS if beam({k: 2 * math.pi * t for k, t in zip(KNOB_NAMES, d)})["power"] < 0.01][:20]
START_TURNS = dict(zip(KNOB_NAMES, OFFSETS[SEED]))
for _k, _t in START_TURNS.items():
    start_at("bench", _k, 2 * math.pi * _t)

# -- scoring --------------------------------------------------------------------------------------------------------------
FULL, HALF, QUARTER = 0.80, 0.40, 0.10  # the reading (fraction of the aligned reference) for 1.0, 0.5 and 0.25
CLEAR = 0.05  # "hands away": both tool points at least this far from every knob at the end
# Keep-out boxes (world, around the laser, both irises (with their levers), the photodiode and each mirror's face): a tool point
# inside one at any time voids the trial (score 0). (centre, half-size)
_KEEP = [(LASER + [0.020, 0, 0], (0.040, 0.025, 0.025)),  # laser module and its lens mount
         (A1 + [0.004, 0.004, -0.008], (0.015, 0.030, 0.035)),  # irises, with their levers
         (A2 + [0.004, 0.004, -0.008], (0.015, 0.030, 0.035)),
         (DET + [0.008, 0, 0], (0.020, 0.025, 0.025)),  # photodiode holder
         (M1 + 0.02 * N1_0, (0.02, 0.02, 0.02)), (M2 + 0.02 * N2_0, (0.02, 0.02, 0.02))]  # the mirrors' faces
# in each mount's own frame (x out of the mirror): its body (plates) and its pedestal post (min corner, max corner)
_KEEP_MOUNT = [((-0.026, -0.022, -0.022), (0.006, 0.022, 0.022)), ((-0.025, -0.030, -0.25), (0.005, 0.030, -0.022))]
_TRACE = os.environ.get("TASK_TRACE")
_EVERY = _env("TRACE_EVERY", 10)
_state = {"void": None, "calls": 0, "best": 0.0}


def _world_knob(name) -> np.ndarray:
    m, k = name.split("_")
    c, yaw = (M1, YAW1) if m == "m1" else (M2, YAW2)
    y, z = KNOBS[k]
    return c + _rz(yaw) @ np.array([KNOB_X, y, z])


def knob_turns(objects) -> dict:
    """Each knob's angle from its aligned zero, in turns."""
    q = objects["bench"].q
    return {k: q[k] / (2 * math.pi) for k in KNOB_NAMES}


def reading(objects) -> float:
    """The photodiode's reading as a fraction of the aligned reference."""
    return beam(objects["bench"].q)["power"]


def _tcps(objects):
    return {a: np.asarray(h["tcp"], float) - np.array([0, 0, TOP]) for a, h in objects["bench"].hands.items()}


def voided(objects) -> str | None:
    """Latched: the first time a tool point was inside a keep-out box (laser, apertures, photodiode, a mirror's face)."""
    if _state["void"] is None:
        for arm, p in _tcps(objects).items():
            for i, (c, h) in enumerate(_KEEP):
                if np.all(np.abs(p - np.asarray(c)) <= np.asarray(h)):
                    _state["void"] = f"{arm} in keep-out {i}"
            for m, c, yaw in (("m1", M1, YAW1), ("m2", M2, YAW2)):
                loc = _rz(-yaw) @ (p - c)
                for lo, hi in _KEEP_MOUNT:
                    if np.all(loc >= lo) and np.all(loc <= hi):
                        _state["void"] = f"{arm} on {m}'s body or post"
    return _state["void"]


def hands_away(objects) -> bool:
    knobs = [_world_knob(k) for k in KNOB_NAMES]
    return all(min(np.linalg.norm(p - k) for k in knobs) >= CLEAR for p in _tcps(objects).values())


def score(objects):
    """0 if a tool point ever entered a keep-out box (laser, apertures, photodiode, mirror faces), or if a hand is still
    on (within 5 cm of) a knob at the end. Otherwise by the photodiode's reading P (fraction of the aligned reference):
    1.0 at P >= 0.80, 0.5 at P >= 0.40, 0.25 at P >= 0.10, else 0."""
    b = beam(objects["bench"].q)
    v = voided(objects)
    _state["calls"] += 1
    if _TRACE and _state["calls"] % _EVERY == 0:
        with open(_TRACE, "a") as f:
            f.write(json.dumps({"call": _state["calls"], "P": round(b["power"], 4), "r1_mm": round(b["r1"] * 1e3, 2),
                                "r2_mm": round(b["r2"] * 1e3, 2), "turns": {k: round(v_, 3) for k, v_ in knob_turns(objects).items()},
                                "void": v, "away": hands_away(objects),
                                "tcp": {a: [round(float(x), 3) for x in p] for a, p in _tcps(objects).items()},
                                "open": {a: round(float(h["open"]), 3) for a, h in objects["bench"].hands.items()}}) + "\n")
    if v is not None or not hands_away(objects):
        return 0.0
    if b["power"] >= FULL:
        return 1.0
    if b["power"] >= HALF:
        return 0.5
    return 0.25 if b["power"] >= QUARTER else 0.0


def success(objects):
    return bool(score(objects) == 1.0)


def distance(objects):
    return float(max(0.0, FULL - reading(objects)))


# -- scripted demo ----------------------------------------------------------------------------------------------------------
# The demo reads the starting knob angles (simulator state, allowed for scripted demos) and turns each knob back to zero:
# the left arm does M1's two knobs, the right arm M2's, each gripping the knob from behind along its axis and turning it
# with the wrist (the arm can roll its gripper through about 240 deg in one grasp here; longer turns take two grasps).

_down_grip = _demo.grip_rotation


def _grip_rotation(yaw=0.0):
    return np.array(yaw, float).reshape(3, 3) if np.ndim(yaw) else _down_grip(yaw)


_demo.grip_rotation = _grip_rotation

GRIP_OPEN = -0.014  # fingers 28 mm apart round the Ø12 mm knob
APPROACH_BACK = 0.04
TCP_PAST = 0.006  # the tool point this far behind the knob's outer face (pad tips 7 mm past it, clear of the plate)
TURN_S = _env("TURN_S", 3.0)  # seconds per 100 deg of turn
ROLL0 = math.radians(_env("ROLL0", 120.0))  # roll of the first grasp; the reachable range here is 60..300 deg
ROLL_MIN, ROLL_MAX = math.radians(65.0), math.radians(295.0)
SLIP = _env("SLIP", 1.015)  # turn this many times the needed angle (tune for slip in the grip)
PROOF = os.environ.get("PROOF", "")  # PROOF=<knob>: one arm only turns that knob by +0.47 turn (a physics check)


def _R(app, roll):
    a = np.asarray(app, float)
    a = a / np.linalg.norm(a)
    c0 = np.cross([0, 0, 1.0], a)
    c0 /= np.linalg.norm(c0)
    c1 = np.cross(a, c0)
    cl = math.cos(roll) * c0 + math.sin(roll) * c1
    return tuple(np.c_[cl, np.cross(a, cl), a].reshape(-1))


def _wp(t, p, f, label, rot):
    return Waypoint(float(t), tuple(float(v) for v in (np.asarray(p) + np.array([0, 0, TOP]))), f, label, rot)


def _reroll(p, app, r0, r1, label):
    """At ``p`` (backed off), roll the gripper from r0 to r1 in steps of at most 40 deg, staying inside the reachable range."""
    n = max(1, int(math.ceil(abs(r1 - r0) / math.radians(40))))
    return [_wp(0.6, p, GRIP_OPEN, label, _R(app, r0 + (r1 - r0) * i / n)) for i in range(1, n + 1)]


def _turn_knob(name, turns, roll):
    """Waypoints that turn knob ``name`` by ``turns`` (positive = counter-clockwise seen from behind the mount), starting
    with the gripper at roll ``roll`` (rad) behind it; returns (waypoints, final roll)."""
    m = name.split("_")[0]
    yaw = YAW1 if m == "m1" else YAW2
    normal_back = _rz(yaw) @ np.array([-1.0, 0, 0])  # from the mount toward the robot
    app = -normal_back  # the gripper points into the mount along the knob's axis
    face = _world_knob(name) + normal_back * (KNOB_L / 2)  # the knob's outer face
    grip = face - app * TCP_PAST
    back = grip - app * APPROACH_BACK
    # measured in the simulator: a gripper roll of +d (about the approach axis, into the mount) turns the knob by +d
    left = 2 * math.pi * turns * SLIP
    out = [_wp(1.5, back, GRIP_OPEN, f"{name}: behind the knob", _R(app, roll))]
    while abs(left) > 1e-3:
        want = ROLL_MIN if left > 0 else ROLL_MAX  # start each grasp at the end of the range that leaves the most turn
        if abs(left) < abs(want - roll) or (left > 0 and roll + left <= ROLL_MAX) or (left < 0 and roll + left >= ROLL_MIN):
            want = roll if ROLL_MIN <= roll + left <= ROLL_MAX else want
        out += _reroll(back, app, roll, want, f"{name}: set the wrist") if abs(want - roll) > 1e-6 else []
        roll = want
        step = max(min(left, ROLL_MAX - roll), ROLL_MIN - roll)
        out += [_wp(1.2, grip, GRIP_OPEN, f"{name}: fingers round the knob", _R(app, roll)),
                _wp(0.6, grip, FINGER_CLOSED, f"{name}: grip", _R(app, roll))]
        n = max(2, int(abs(step) / math.radians(20)))
        for i in range(1, n + 1):
            out.append(_wp(TURN_S * abs(step) / math.radians(100) / n, grip, FINGER_CLOSED, f"{name}: turn",
                           _R(app, roll + step * i / n)))
        roll += step
        left -= step
        out += [_wp(0.5, grip, GRIP_OPEN, f"{name}: let go", _R(app, roll)),
                _wp(1.0, back, GRIP_OPEN, f"{name}: back off", _R(app, roll))]
    return out, roll


def _start(arm):
    """From the ready pose to a point behind the arm's mirror, the gripper already pointing along its knobs."""
    m = "m1" if arm == "left" else "m2"
    yaw = YAW1 if m == "m1" else YAW2
    nb = _rz(yaw) @ np.array([-1.0, 0, 0])
    c = (M1 if m == "m1" else M2) + nb * 0.12 + np.array([0, 0, 0.04])
    return [_wp(3.0, c, GRIP_OPEN, f"{arm}: toward its mirror", _R(-nb, ROLL0))]


def _finish(arm, roll):
    m = "m1" if arm == "left" else "m2"
    yaw = YAW1 if m == "m1" else YAW2
    nb = _rz(yaw) @ np.array([-1.0, 0, 0])
    c = (M1 if m == "m1" else M2) + nb * 0.15 + np.array([0, 0, 0.06])
    return [_wp(2.0, c, GRIP_OPEN, f"{arm}: hands away", _R(-nb, roll)),
            _wp(6.0, c, GRIP_OPEN, f"{arm}: hold clear while the meter is read", _R(-nb, roll))]


def demo():
    if PROOF:
        arm = "left" if PROOF.startswith("m1") else "right"
        w, roll = _turn_knob(PROOF, 0.47, ROLL0)
        return {arm: _start(arm) + w + _finish(arm, roll)}
    plan = {}
    for arm, m in (("left", "m1"), ("right", "m2")):
        w, roll = _start(arm), ROLL0
        for i, k in enumerate(("up", "side")):
            name = f"{m}_{k}"
            if i:  # between knobs: wrist back to its starting roll, out to the start point, as at the first knob
                yaw = YAW1 if m == "m1" else YAW2
                nb = _rz(yaw) @ np.array([-1.0, 0, 0])
                prev = f"{m}_up"
                back = _world_knob(prev) + nb * (KNOB_L / 2 + TCP_PAST + APPROACH_BACK)
                w += _reroll(back, -nb, roll, ROLL0, f"{arm}: wrist round")
                w += _start(arm)
                roll = ROLL0
            more, roll = _turn_knob(name, -START_TURNS[name], roll)
            w += more
        plan[arm] = w + _finish(arm, roll)
    return plan
