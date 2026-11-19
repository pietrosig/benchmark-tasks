"""On the station table, a dinner plate on a non-slip mat holds one cooked sausage, lying left to right across its middle. A dinner fork lies to the left of the plate, its handle on a small rest; a serrated table knife stands edge
down in a slotted rest to the right.

Task: cut three bite-size pieces (1.5 to 2.5 cm long) off the sausage with the knife and fork, keeping everything on
the plate, then lay the knife and fork down on the plate. The link is round and slick on the glazed plate: the fork's
tines pin it while the knife saws through it beside them.

Run it:  inspect-robots run --embodiment yam_arena --policy <policy> -E task=task.py \\
           --instruction "Cut three bite-size pieces, 1.5 to 2.5 cm long, off the sausage with the knife and fork, keeping everything on the plate, then lay the knife and fork down." \\
           --scorer task_score -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 --max-steps 2400
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from foodkit import (Body, allow_matrix_grips, as_np, box, cyl, glue, grip, grip_off, grip_on, on_reset,  # noqa: E402
                     on_step, pose, quat_of, robot_contacts, rotation, set_force, set_glue)
from yam_arena.constants import FINGER_CLOSED, FINGER_JOINTS, FINGER_OPEN, TCP_OFFSET  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

WHITE, MAT_C, SKIN, MEAT, END_C = (0.95, 0.95, 0.93), (0.16, 0.17, 0.18), (0.58, 0.32, 0.17), (0.88, 0.60, 0.52), (0.45, 0.24, 0.13)
STEEL, KNIFE_H, WOOD = (0.80, 0.81, 0.83), (0.12, 0.12, 0.13), (0.55, 0.38, 0.22)

# -- the plate: a 26 cm glazed porcelain dinner plate on a non-slip mat (fixed: the mat holds it) ----------------------
PX, PY = 0.30, 0.0
MAT_T, FLOOR_T = 0.003, 0.010
PF = TOP + MAT_T + FLOOR_T  # the plate's floor
R_FLOOR, R_PLATE, RIM_H = 0.095, 0.130, 0.016


def _plate() -> Body:
    shapes = [box((0, 0, MAT_T / 2), (0.32, 0.32, MAT_T), MAT_C),
              cyl((0, 0, MAT_T + FLOOR_T / 2), R_FLOOR, FLOOR_T, "Z", WHITE)]
    n = 28
    for i in range(n):  # the rim: a ring of panels rising from the floor's edge to the plate's edge
        a = 2 * math.pi * i / n
        r0, r1 = R_FLOOR, R_PLATE
        slope = math.atan2(RIM_H, r1 - r0)
        length = math.hypot(r1 - r0, RIM_H)
        rm = (r0 + r1) / 2
        rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        ry = np.array([[math.cos(-slope), 0, math.sin(-slope)], [0, 1, 0], [-math.sin(-slope), 0, math.cos(-slope)]])
        shapes.append(box((rm * math.cos(a), rm * math.sin(a), MAT_T + FLOOR_T + RIM_H / 2 - 0.002),
                          (length, 2 * math.pi * R_PLATE / n * 1.05, 0.004), WHITE, quat=quat_of(rz @ ry)))
    return Body.build("plate", (PX, PY, TOP), 1.0, shapes, kinematic=True, friction=(0.3, 0.25))


# -- the sausage: a fully cooked link, 3 cm across and 13 cm long, as 26 slabs 5 mm long glued end to end ---------------
R, SL, N = 0.015, 0.010, 13
SX, Y_R = PX, PY - 0.065  # its axis x, its right end's y: it lies left to right across the plate
MASS = 0.080


def _slab(i: int) -> Body:
    sh = [cyl((0, 0, 0), R, SL, "Y", SKIN), box((0, 0, -R + 0.001), (0.012, SL, 0.002), SKIN)]  # it lies on a flat
    for s in (-1, 1):  # the meat shows on a cut face (inside the skin)
        c = END_C if (i == 0 and s < 0) or (i == N - 1 and s > 0) else MEAT
        sh.append(cyl((0, s * (SL / 2 + 0.0001), 0), R - 0.0012, 0.0002, "Y", c, collide=False))
    return Body.build(f"slab_{i}", (SX, Y_R + (i + 0.5) * SL, PF + R + 0.0003), MASS / N, sh,
                      friction=(0.5, 0.45), damping=(0.05, 0.2))


SLABS = [_slab(i) for i in range(N)]
SLAB_NAMES = [s.name for s in SLABS]
for k in range(1, N):  # glue line k: between slab k-1 and slab k, k x 5 mm from the right end
    glue(f"cut_{k}", f"slab_{k - 1}", f"slab_{k}", (SX, Y_R + k * SL, PF + R + 0.0003))

# -- the fork: a stainless dinner fork, 19 cm, 4 tines 3.5 cm long. Its frame: origin at the middle of the tines' tips,
# x along the fork toward the handle, y across the tines, z out of its front face -----------------------------------------
TINE_L, TINE_Y = 0.035, (-0.0075, -0.0025, 0.0025, 0.0075)
FORK_GRIP_X = float(os.environ.get("FORK_GRIP", 0.130))  # where the demo hand holds it: on the handle, 13 cm from the tips
FORK_MASS, FORK_COM = 0.045, (0.115, 0.0, 0.0)


TINE_BEND = math.radians(20.0)  # the tines curve down 20 degrees from the handle's line, as a dinner fork's do
TINE_DIR_F = np.array([-math.cos(TINE_BEND), 0.0, -math.sin(TINE_BEND)])  # the way the tines point, in the fork's frame
HEAD_UP = TINE_L * math.sin(TINE_BEND)  # the head and handle sit this far above the tips' line


def _fork(pos, quat) -> Body:
    q_t = quat_of(np.array([[math.cos(TINE_BEND), 0, -math.sin(TINE_BEND)], [0, 1, 0],
                            [math.sin(TINE_BEND), 0, math.cos(TINE_BEND)]]))
    mid = -TINE_DIR_F * TINE_L / 2
    sh = [box((mid[0], y, mid[2]), (TINE_L, 0.0018, 0.0015), STEEL, quat=q_t) for y in TINE_Y]
    hx = TINE_L * math.cos(TINE_BEND)
    sh += [box((hx + 0.006, 0, HEAD_UP), (0.012, 0.024, 0.002), STEEL),
           box((hx + 0.036, 0, HEAD_UP), (0.042, 0.009, 0.003), STEEL),
           box((hx + 0.105, 0, HEAD_UP), (0.098, 0.013, 0.0045), STEEL)]
    return Body.build("fork", pos, FORK_MASS, sh, friction=(0.5, 0.4), com=FORK_COM, quat0=quat, ignore=tuple(SLAB_NAMES))


# -- the knife: a stainless serrated table knife, 22 cm: blade 11 cm x 2 cm, handle 11 cm. Its frame: origin on the edge
# line where the blade meets the handle, x toward the tip, z up toward the spine ---------------------------------------
BLADE_L, BLADE_H, BLADE_T = 0.105, 0.020, 0.0018
HANDLE_L, HANDLE_Z, HANDLE_W, HANDLE_H = 0.135, 0.024, 0.011, 0.020
KNIFE_MASS, KNIFE_COM = 0.075, (0.020, 0.0, 0.012)
KNIFE_GRIP = float(os.environ.get("KNIFE_GRIP", 0.120))  # the hand holds the handle this far behind the blade


def _knife(pos) -> Body:
    sh = [box((BLADE_L / 2, 0, BLADE_H / 2), (BLADE_L, BLADE_T, BLADE_H), STEEL),
          box((-HANDLE_L / 2, 0, HANDLE_Z), (HANDLE_L, HANDLE_W, HANDLE_H), STEEL)]
    return Body.build("knife", pos, KNIFE_MASS, sh, friction=(1.0, 0.9), com=KNIFE_COM, ignore=tuple(SLAB_NAMES))


# where they start: the knife edge down in a slotted rest right of the plate; the fork lying left of the plate, tines on
# the table, its handle on a 12 mm rest
REST_X, REST_Y, REST_BASE = 0.245, -0.205, 0.004
KNIFE0 = (REST_X, REST_Y, TOP + REST_BASE + 0.0004)
FORK_X, FORK_TIP_Y, FORK_REST_FX, FORK_REST_H = 0.300, 0.150, 0.165, 0.0115  # the fork along y, tines toward the plate
_fork_pitch = math.atan2(FORK_REST_H + 0.00225 - 0.001 - HEAD_UP, FORK_REST_FX)  # tips on the table, handle on the rest
_c, _s = math.cos(_fork_pitch), math.sin(_fork_pitch)
FORK_Q0 = quat_of(np.array([[0, -1, 0], [_c, 0, -_s], [_s, 0, _c]]))  # x toward the handle (+y, up)
FORK0 = (FORK_X, FORK_TIP_Y, TOP + 0.0009)
FORK_REST_Y = FORK_TIP_Y + FORK_REST_FX * _c


def _knife_rest() -> Body:
    sh = [box((0, 0, REST_BASE / 2), (0.11, 0.05, REST_BASE), WOOD)]
    sh += [box((0, s * 0.006, REST_BASE + 0.012), (0.11, 0.008, 0.024), WOOD) for s in (-1, 1)]
    return Body.build("knife_rest", (REST_X + 0.062, REST_Y, TOP), 1.0, sh, kinematic=True, friction=(0.4, 0.3))


def _fork_rest() -> Body:
    return Body.build("fork_rest", (FORK_X, FORK_REST_Y, TOP), 1.0,
                      [box((0, 0, FORK_REST_H / 2), (0.05, 0.016, FORK_REST_H), WOOD)], kinematic=True, friction=(0.5, 0.4))


PLATE = _plate()
FORK = _fork(FORK0, FORK_Q0)
KNIFE = _knife(KNIFE0)
OBJECTS = [PLATE, _knife_rest(), _fork_rest(), *SLABS, FORK, KNIFE]
for i in range(N):  # the fork's hold on each slab: switched on when its tines go in
    grip(f"tines_{i}", "fork", f"slab_{i}", slide_cap=3.0, slide_max=0.015)

# -- the rules (estimates; see README) ---------------------------------------------------------------------------------
K_PIERCE, PIERCE_N = 800.0, 2.5  # the skin pushes the tines back (N per m of indent) until 2.5 N: then they go in
TINE_MAX_DEG = 55.0  # tines more than this off the skin's normal skid instead of going in
TINE_MU = 0.15  # tine tips on skin: friction along the skin before they bite
SKID_N = 1.0  # tines too slanted to the skin skid off it, pushing no more than this
TEAR_SLIDE, TEAR_DEG = 0.008, 35.0
REPIERCE_STEPS = 20  # after tearing out, the tines cannot go in again for 1 s  # tines out: drawn 8 mm back (a pull over the 3 N hold), or levered 35 deg over
K_CUT, PRESS_CUT_N, SAW_MIN_N = 600.0, 8.0, 1.0  # uncut food pushes the edge back (N/m) up to 8 N: pressed harder, it
# goes through; sawing cuts only while the edge presses at least 2 N
THROUGH_GAP = 0.004  # the edge within 4 mm of the plate under the food: the last of it parts
RATE = 0.30  # m of depth cut per m sawn along the blade
ON_LINE, KERF = 0.005, 0.010
START_DEPTH = 0.006  # a new cut starts only from the top of the skin
DRAG_K, DRAG_MAX = 12.0, 1.2  # the edge drags the food along with the stroke (N per m/s of blade speed, at most 1.2 N)
BLADE_ALIGN_DEG = 35.0

NUDGE_N, NUDGE_STEPS = 0.07, 3  # a freed face is pushed 0.07 N apart for 0.15 s (the blade's thickness in the kerf)
PIECE_MIN, PIECE_MAX = 2, 2  # slabs in a bite-size piece: a cut 1.5 to 2.5 cm from the end lands on the 2 cm line
OFF_PLATE_R = 0.120  # a slab's middle further than this from the plate's centre is off the plate
CLEAR = 0.03
LAID_TOP = PF + 0.050

_TRACE = os.environ.get("TASK_TRACE")
_DBG = (_TRACE + ".dbg") if (_TRACE and os.environ.get("CUTDBG")) else None


def _fresh():
    return {"cooldown": 0, "nudge": [], "active": None, "cut": [0.0] * N, "through": [False] * N, "pinned_cut": [False] * N, "saw_s": {}, "pierced": None,
            "pierce_dir": None, "anchor": None, "touched": [], "off": False, "tcps": None, "open": None,
            "prev_knife": None, "steps": 0, "n_pierce": 0, "n_tear": 0, "events": []}


EP = _fresh()


@on_reset
def _reset():
    EP.clear()
    EP.update(_fresh())


def _tcps(base):
    o = as_np(base.scene.env_origins[0])
    out, opening = [], []
    for arm in ("left_robot", "right_robot"):
        rb = base.scene[arm]
        i = next(j for j, n in enumerate(rb.body_names) if n == "gripper" or n.endswith("/gripper"))
        g = as_np(rb.data.body_link_pose_w)[0][i]
        out.append(g[:3] - o + rotation(g[3:7]) @ np.array(TCP_OFFSET))
        f = rb.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        opening.append(float(np.clip(as_np(rb.data.joint_pos)[0][f].mean() / FINGER_OPEN, 0, 1)))
    return out, opening


def _slabs(base):
    return [pose(base, n) for n in SLAB_NAMES]


def _log(**kw):
    if _TRACE:
        with open(_TRACE, "a") as fh:
            fh.write(json.dumps(kw) + "\n")


def _fork_step(base, sl):
    """The tines: pushed into the skin they are held back until 4 N, then go in and hold (a joint); drawn out or levered
    over they come out."""
    fp, fr, _, _ = pose(base, "fork")
    tine_dir = fr @ TINE_DIR_F  # the way the tines point
    tips = [fp + fr @ np.array([0.0, y, 0.0]) for y in TINE_Y]
    if EP["pierced"] is not None:
        j = EP["pierced"]
        sp, sr = sl[j][0], sl[j][1]
        anchor_w = sp + sr @ EP["anchor"][1]
        tip_w = fp + fr @ EP["anchor"][0]
        drawn = float(np.dot(tip_w - anchor_w, -tine_dir))
        lever = math.degrees(math.acos(float(np.clip(np.dot(tine_dir, EP["pierce_dir"]), -1, 1))))
        if drawn > TEAR_SLIDE or lever > TEAR_DEG:
            grip_off(f"tines_{j}")
            EP["pierced"] = None
            EP["n_tear"] += 1
            EP["cooldown"] = REPIERCE_STEPS
            EP["events"].append(("out", EP["steps"], round(drawn * 1000, 1), round(lever, 1)))
        set_force(base, "fork", np.zeros(3))
        return {}
    best = (0.0, None, None, None)
    for t_i, tip in enumerate(tips):
        for j, (sp, sr, _, _) in enumerate(sl):
            v = sr.T @ (tip - sp)
            if abs(v[1]) <= SL / 2 + 1e-4:
                rr = math.hypot(v[0], v[2])
                depth = R - rr
                if depth > best[0]:
                    n = sr @ np.array([v[0], 0.0, v[2]]) / max(rr, 1e-6)  # the skin's outward normal there
                    best = (depth, j, t_i, n)
    depth, j, t_i, n = best
    if j is None:
        set_force(base, "fork", np.zeros(3))
        return {}
    into = math.degrees(math.acos(float(np.clip(np.dot(tine_dir, -n), -1, 1))))  # tines off the skin's normal
    push = min(K_PIERCE * depth, PIERCE_N if into <= TINE_MAX_DEG else SKID_N)
    if EP["cooldown"] > 0:  # just torn out: the tines leave their holes before they can bite again
        EP["cooldown"] -= 1
    elif K_PIERCE * depth >= PIERCE_N and into <= TINE_MAX_DEG:
        sp, sr = sl[j][0], sl[j][1]
        tip_local_f = np.array([0.0, TINE_Y[t_i], 0.0])
        tip_w = fp + fr @ tip_local_f
        r_slide = np.column_stack([-TINE_DIR_F, [0.0, 1.0, 0.0], np.cross(-TINE_DIR_F, [0.0, 1.0, 0.0])])  # x: out
        grip_on(f"tines_{j}", tip_local_f, r_slide, sr.T @ (tip_w - sp), sr.T @ fr @ r_slide)
        EP["pierced"], EP["pierce_dir"], EP["anchor"] = j, tine_dir, (tip_local_f, sr.T @ (tip_w - sp))
        EP["n_pierce"] += 1
        EP["events"].append(("in", EP["steps"], j, round(into, 1)))
        set_force(base, "fork", np.zeros(3))
        return {}
    # the skin holds the tips out: a push along its normal, and along the skin no more than tine-on-skin friction
    t_part = tine_dir - np.dot(tine_dir, n) * n
    tn = float(np.linalg.norm(t_part))
    f_food = -n * push * abs(float(np.dot(tine_dir, n)))
    if tn > 1e-6:
        f_food = f_food + t_part / tn * min(tn * push, TINE_MU * push)
    set_force(base, "fork", -f_food)
    return {j: f_food}


def _seg_dist(p, a, b) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _knife_step(base, sl):
    kp, kr, _, _ = pose(base, "knife")
    kx, ky = kr[:, 0], kr[:, 1]
    prev = EP["prev_knife"]
    EP["prev_knife"] = kp.copy()
    forces = {}
    push = 0.0
    best = None
    act = EP.get("active")
    if act is not None and not EP["through"][act]:  # the blade is in this cut's kerf: it stays in it until lifted out
        a, b = sl[act - 1][0], sl[act][0]
        bk = (a + b) / 2
        if float(np.dot(kp - bk, [0, 0, 1.0])) < R + 0.004 and abs(float(np.dot(ky, bk - kp))) < 0.015:
            best = (act, float(np.dot(ky, bk - kp)), bk, (0, 0.0))
        else:
            EP["active"] = None
    for k in range(1, N):
        if best is not None and best[3] == (0, 0.0):
            break
        if EP["through"][k]:
            continue
        a, b = sl[k - 1][0], sl[k][0]
        ax = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        if abs(float(np.dot(ky, ax))) < math.cos(math.radians(BLADE_ALIGN_DEG)):
            continue
        bk = (a + b) / 2
        s = float(np.dot(ky, bk - kp))
        started = EP["cut"][k] > 0.002
        lim = KERF if started else ON_LINE
        rank = (0 if started else 1, abs(s))  # a cut already started holds the blade in its kerf
        if abs(s) <= lim and (best is None or rank < best[3]):
            best = (k, s, bk, rank)
    held = any(_seg_dist(t, kp + HANDLE_Z * kr[:, 2], kp + HANDLE_Z * kr[:, 2] - HANDLE_L * kx) < 0.03
               for t in (EP["tcps"] or []))
    slow = prev is None or float(np.linalg.norm(kp - prev)) * 20.0 < 0.5
    if best is not None and held and slow:  # only a knife in a hand, moved at hand speed, cuts (not a dropped one)
        k, s, bk, _ = best
        along = float(np.dot(bk - kp, kx))
        if -0.010 < along < BLADE_L + 0.010:
            s0 = float(np.clip(along, 0, BLADE_L))
            edge = kp + s0 * kx
            depth = min((bk[2] + R) - edge[2], 2 * R + 0.002)
            cut = EP["cut"][k]
            if depth > 0 and cut < 0.0005 and depth > START_DEPTH and EP.get("active") != k:
                depth = -1.0  # an edge already deep in the food beside a finished cut cannot start a new cut from inside
            if depth > 0:
                EP["active"] = k  # the edge sits in the groove it presses into the skin: it stays on this line
                over = depth - cut
                if prev is not None:
                    saw = abs(float(np.dot(kp - prev, kx)))
                    if K_CUT * over >= SAW_MIN_N:
                        cut = min(cut + RATE * saw, depth)
                        EP["saw_s"][k] = EP["saw_s"].get(k, 0.0) + saw
                    blade_v = float(np.dot(kp - prev, kx)) * 20.0
                    if over > 0.0005:
                        h = kx - np.dot(kx, [0, 0, 1.0]) * np.array([0, 0, 1.0])
                        h = h / max(np.linalg.norm(h), 1e-9)
                        forces[k] = h * float(np.clip(DRAG_K * blade_v, -DRAG_MAX, DRAG_MAX))
                if K_CUT * (depth - cut) > PRESS_CUT_N:  # pressed hard enough, it goes through without sawing
                    cut = depth - PRESS_CUT_N / K_CUT
                cut = max(cut, EP["cut"][k])
                EP["cut"][k] = cut
                if cut > 0.0:
                    EP["active"] = k  # the edge is in this kerf now
                if cut >= 2 * R - THROUGH_GAP:
                    EP["through"][k] = True
                    EP["pinned_cut"][k] = EP["pierced"] is not None
                    set_glue(f"cut_{k}", False)
                    EP["nudge"].append([k, NUDGE_STEPS])
                    EP["events"].append(("through", EP["steps"], k, EP["pierced"]))
                push = min(K_CUT * max(depth - cut, 0.0), PRESS_CUT_N)
                if push > 0:  # the edge presses the food down onto the plate
                    forces[k] = forces.get(k, np.zeros(3)) + np.array([0, 0, -push])
    set_force(base, "knife", np.array([0, 0, push]), at_local=(-KNIFE_GRIP, 0, HANDLE_Z))  # at the hand: it does not turn the knife in the grip
    if _DBG and best is not None:
        k_, s_, bk_, _ = best
        al = float(np.dot(bk_ - kp, kx))
        e = kp + float(np.clip(al, 0, BLADE_L)) * kx
        with open(_DBG, "a") as fh:
            fh.write(json.dumps({"st": EP["steps"], "k": k_, "s": round(s_ * 1000, 1), "al": round(al * 1000),
                                 "d": round(((bk_[2] + R) - e[2]) * 1000, 1), "cut": round(EP["cut"][k_] * 1000, 1),
                                 "act": EP.get("active"), "push": round(push, 2),
                                 "yaw": round(math.degrees(math.atan2(kx[1], kx[0])), 1)}) + "\n")
    return forces, push, best


def _pieces():
    """Groups of slabs held together (glue on), right end first, as lists of slab indices."""
    groups, cur = [], [0]
    for k in range(1, N):
        if EP["through"][k]:
            groups.append(cur)
            cur = [k]
        else:
            cur.append(k)
    groups.append(cur)
    return groups


def _valid_pieces():
    out = []
    groups = _pieces()
    rest = max(groups, key=len)
    for g in groups:
        if g is rest or not (PIECE_MIN <= len(g) <= PIECE_MAX):
            continue
        cuts = [k for k in (g[0], g[-1] + 1) if 1 <= k <= N - 1]
        if all(EP["pinned_cut"][k] for k in cuts):
            out.append(g)
    return out


@on_step
def _follow(env):
    base = env.unwrapped
    EP["steps"] += 1
    sl = _slabs(base)
    EP["tcps"], EP["open"] = _tcps(base)
    fork_f = _fork_step(base, sl)
    knife_f, push, best = _knife_step(base, sl)
    nudge = {}
    for item in EP["nudge"]:  # the blade left in a finished kerf parts the two cut faces a few mm
        k, left_steps = item
        if left_steps > 0:
            ax = sl[k][0] - sl[k - 1][0]
            ax = ax / max(np.linalg.norm(ax), 1e-9) * NUDGE_N
            nudge[k - 1] = nudge.get(k - 1, np.zeros(3)) - ax
            nudge[k] = nudge.get(k, np.zeros(3)) + ax
            item[1] -= 1
    for i, n in enumerate(SLAB_NAMES):  # forces from the utensils act at the top of the skin: an unpinned link rolls
        f = fork_f.get(i, np.zeros(3)) + knife_f.get(i, np.zeros(3)) + nudge.get(i, np.zeros(3))
        sp, sr = sl[i][0], sl[i][1]
        set_force(base, n, f, at_local=sr.T @ np.array([0, 0, R]))
    touched = robot_contacts(base, SLAB_NAMES)
    if touched:
        if not EP["touched"]:
            EP["events"].append(("touch", EP["steps"], touched))
        EP["touched"] = touched
    for p, _, _, _ in sl:
        if math.hypot(p[0] - PX, p[1] - PY) > OFF_PLATE_R or p[2] < PF - 0.005:
            if not EP["off"]:
                EP["events"].append(("off", EP["steps"]))
            EP["off"] = True
    if EP["steps"] % 5 == 0:
        fp, fr, _, _ = pose(base, "fork")
        kp, kr, _, _ = pose(base, "knife")
        _log(t=round(EP["steps"] / 20, 2), cut=[round(c * 1000, 1) for c in EP["cut"]],
             through=[k for k in range(N) if EP["through"][k]], pierced=EP["pierced"],
             fork=np.round(fp * 1000).astype(int).tolist(), knife=np.round(kp * 1000).astype(int).tolist(),
             s0=np.round(sl[0][0] * 1000).astype(int).tolist(), s25=np.round(sl[-1][0] * 1000).astype(int).tolist(),
             line=None if best is None else best[0], push=round(push, 2), touched=bool(EP["touched"]), off=EP["off"],
             tcpL=np.round(EP["tcps"][0] * 1000).astype(int).tolist(), tcpR=np.round(EP["tcps"][1] * 1000).astype(int).tolist(),
             open=[round(o, 2) for o in EP["open"]], pieces=[len(g) for g in _pieces()], valid=len(_valid_pieces()),
             events=EP["events"][-4:], score=_points(None))


# -- scoring --------------------------------------------------------------------------------------------------------

def _laid(state, others_tcps) -> bool:
    """A utensil laid down on the plate: its middle over the plate, lying (its highest point low), not in the food, no
    gripper within 3 cm of it."""
    c = (state.low + state.high) / 2
    if math.hypot(c[0] - PX, c[1] - PY) > R_PLATE or state.high[2] > LAID_TOP or state.low[2] < TOP - 0.01:
        return False
    lo, hi = state.low - CLEAR, state.high + CLEAR
    return not any(np.all(t >= lo) and np.all(t <= hi) for t in others_tcps)


def _points(objects) -> float:
    if EP["touched"] or EP["off"]:
        return 0.0
    n = min(len(_valid_pieces()), 3)
    if n < 3 or objects is None:
        return 0.25 * n
    tcps = EP["tcps"] or []
    opened = EP["open"] and min(EP["open"]) > 0.5
    if (EP["pierced"] is None and opened and _laid(objects["fork"], tcps) and _laid(objects["knife"], tcps)):
        return 1.0
    return 0.75


def score(objects) -> float:
    """0 if a gripper ever touched the sausage or any of it left the plate. Otherwise 0.25 per bite-size piece (1.5 to
    2.5 cm, cut fully off with the fork in the sausage during the cut that freed it), at most 0.75; 1.0 with three such
    pieces and the knife and fork both let go, lying on the plate, grippers open and 3 cm clear."""
    return _points(objects)


def success(objects) -> bool:
    if _points(objects) < 1.0:
        return False
    return all(objects[n].resting(0.01, 0.5) for n in ["fork", "knife", *SLAB_NAMES])


# -- scripted demo -------------------------------------------------------------------------------------------------

def _rz(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


DOWN = grip_rotation(0.0)  # straight down, fingers closing along world y
FORK_TILT = math.radians(float(os.environ.get("FORK_TILT", 40)))  # the fork's handle above horizontal while it pins
KNIFE_TILT = math.radians(float(os.environ.get("KNIFE_TILT", 2)))  # the knife's tip below its handle while it saws
TOPZ = PF + 2 * R
PIN_Y = float(os.environ.get("PIN_Y", 0.040))
FORK_FIX = np.array([0.0, 0.0, 0.0])  # aim correction for how the fork sits in the hand
CUTS = [int(c) for c in os.environ.get("CUTS", "2,4,6").split(",") if c]
PIERCE_PUSH = float(os.environ.get("PIERCE_PUSH", 0.020))
SAW = 0.025
LEVER_DEG = 40.0
LAY_F_DX, LAY_K_DX = 0.065, 0.085  # where the fork (far side) and the knife (near rim) are laid
STROKE_S = 0.8
LEAD_S = 3.0


def _w(t, pos, finger, label, rot):
    return Waypoint(float(t), tuple(float(v) for v in pos), float(finger), label, rot)


def _lead(arm, pos, rot, label, finger):
    import itertools

    from yam_arena import ik, station
    from yam_arena.constants import READY

    best = None
    for seed in itertools.product((-0.5, 0.0, 0.5), (0.8, 1.3), (0.4, 1.0), (-0.8, 0.0, 0.8), (-1.4, 0.0, 1.4), (0.0,)):
        q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), np.array(seed),
                             base_pos=station.BASES[arm], iters=200)
        if pe < 0.001 and re < math.radians(0.5):
            room = float(np.minimum(q - ik.LOWER, ik.UPPER - q).min())
            if best is None or room > best[1]:
                best = (q, room)
    if best is None:
        raise ValueError(f"no IK for the {arm} arm at {pos}")
    q0, q1 = np.array(READY, float), best[0]
    out = []
    for k in range(1, 16):
        p, r = ik.tcp_pose(q0 + (q1 - q0) * k / 15, station.BASES[arm])
        out.append(Waypoint(LEAD_S / 15, tuple(float(v) for v in p), finger, label, r))
    return out


# the fork in the hand: it lies along y (handle toward the left arm) and is picked with the fingers closing across its
# handle (along x), the hand pitched PICK_PITCH so that the fork sits in the hand inclined; then the hand pitches the
# handle up by HAND_TILT, so the handle stands FORK_TILT above horizontal: pitches are about the axis z x h (h: the
# handle's horizontal direction, +y), a rotation by -t raising the handle by t
H_DIR = np.array([0.0, 1.0, 0.0])
A_AX = np.cross([0.0, 0.0, 1.0], H_DIR)
G_FORK = grip_rotation(-math.pi / 2)
HAND_TILT = math.radians(float(os.environ.get("HAND_TILT", 15)))
PICK_PITCH = HAND_TILT + _fork_pitch - FORK_TILT  # so that the fork ends at FORK_TILT with the hand at HAND_TILT


def _rot(axis, t):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def _fork_x(tilt):
    return _rot(A_AX, -tilt) @ H_DIR  # the fork's x axis (tips toward the handle) with the handle tilt up


def _fork_grip(tilt):
    """From the tips to where the hand holds the handle, with the handle tilt up (world)."""
    return FORK_GRIP_X * _fork_x(tilt) + HEAD_UP * (_rot(A_AX, -tilt) @ np.array([0.0, 0.0, 1.0]))


def _fork_tines(tilt):
    r = _rot(A_AX, -tilt)
    return -math.cos(TINE_BEND) * (r @ H_DIR) - math.sin(TINE_BEND) * (r @ np.array([0.0, 0.0, 1.0]))


def _left():
    tips0 = np.array(FORK0)
    grip0 = tips0 + _fork_grip(_fork_pitch) + np.array([0, 0, 0.001])
    r_pick = _rot(A_AX, -PICK_PITCH) @ G_FORK
    r_tilt = _rot(A_AX, -HAND_TILT) @ G_FORK
    out = _lead("left", grip0 + [0, 0, 0.05], r_pick, "over the fork's handle", FINGER_OPEN)
    out += [_w(1.0, grip0, FINGER_OPEN, "down round the handle", r_pick),
            _w(0.8, grip0, FINGER_CLOSED, "grip the fork", r_pick),
            _w(0.3, grip0, FINGER_CLOSED, "settle", r_pick),
            _w(1.0, grip0 + [0, 0, 0.07], FINGER_CLOSED, "lift it", r_pick)]
    g = _fork_grip(FORK_TILT)
    d = _fork_tines(FORK_TILT)  # the way the tines point
    pin = np.array([SX, PIN_Y, TOPZ]) + FORK_FIX  # aimed off by how the fork sits in the hand (from a trial)
    out += [_w(2.0, pin + [0, 0, 0.03] + g, FINGER_CLOSED, "tines over the sausage, the fork tilted", r_tilt),
            _w(1.5, pin + g, FINGER_CLOSED, "tines on the skin", r_tilt),
            _w(1.4, pin + g + d * PIERCE_PUSH, FINGER_CLOSED, "push them in", r_tilt),
            _w(0.8, pin + g + d * 0.008, FINGER_CLOSED, "ease off: hold it lightly", r_tilt)]
    return out, pin + g + d * 0.008, r_tilt, d


def _rx(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])


def _knife_tcp(edge_pt, tilt):
    """The right TCP on the knife's handle, the knife pointing +x (away from the arms, handle toward them), its tip tilt
    below its handle, with the blade's middle on the edge at edge_pt (world)."""
    r = _ry(tilt)
    kx, kz = r @ np.array([1.0, 0, 0]), r @ np.array([0, 0, 1.0])
    origin = edge_pt - (BLADE_L / 2) * kx
    return origin + (-KNIFE_GRIP) * kx + (HANDLE_Z + 0.003) * kz, r @ DOWN


def _right():
    pick = np.array(KNIFE0) + np.array([-KNIFE_GRIP, 0, HANDLE_Z + 0.003])
    out = _lead("right", pick + [0, 0, 0.05], DOWN, "over the knife's handle", FINGER_OPEN)
    out += [_w(1.0, pick, FINGER_OPEN, "down round the handle", DOWN),
            _w(0.8, pick, FINGER_CLOSED, "grip the knife", DOWN),
            _w(0.3, pick, FINGER_CLOSED, "settle", DOWN),
            _w(1.2, pick + [0, 0, 0.08], FINGER_CLOSED, "lift it out of the rest", DOWN)]
    t, rot = _knife_tcp(np.array([SX, Y_R + 0.02, TOPZ + 0.05]), 0.0)
    out.append(_w(1.6, t, FINGER_CLOSED, "over the sausage's right end", rot))
    return out


def _saw(out, y, kt):
    kx = _ry(KNIFE_TILT) @ np.array([1.0, 0, 0])
    top_pt = np.array([SX, y, TOPZ])
    t, rot = _knife_tcp(top_pt + [0, 0, 0.02], KNIFE_TILT)
    out.append(_w(1.4, t, FINGER_CLOSED, f"blade over the cut {round((y - Y_R) * 100, 1)} cm from the right end", rot))
    t, _ = _knife_tcp(top_pt, KNIFE_TILT)
    out.append(_w(0.6, t, FINGER_CLOSED, "edge on the skin", rot))
    if BYPASS == "press":  # bypass test: press straight down, no sawing
        t, _ = _knife_tcp(np.array([SX, y, PF - 0.004]), KNIFE_TILT)
        out.append(_w(3.0, t, FINGER_CLOSED, "press straight down", rot))
        t, _ = _knife_tcp(top_pt + [0, 0, 0.02], KNIFE_TILT)
        out.append(_w(0.8, t, FINGER_CLOSED, "lift the knife", rot))
        return
    depth, sign = 0.0, 1.0
    while depth < 2 * R + 0.016:
        depth = min(depth + RATE * 2 * SAW * 0.5, 2 * R + 0.016)
        t, _ = _knife_tcp(top_pt - [0, 0, min(depth + 0.004, 2 * R - 0.002)], KNIFE_TILT)  # never into the plate
        out.append(_w(STROKE_S, t + sign * SAW * kx, FINGER_CLOSED, "saw, lowering the edge as it cuts", rot))
        sign = -sign
    t, _ = _knife_tcp(np.array([SX, y, PF + 0.002]), KNIFE_TILT)
    out.append(_w(STROKE_S, t + sign * SAW * kx, FINGER_CLOSED, "one more stroke down to the plate", rot))
    out.append(_w(STROKE_S, t - sign * SAW * kx, FINGER_CLOSED, "and back", rot))
    t, _ = _knife_tcp(top_pt + [0, 0, 0.02], KNIFE_TILT)
    out.append(_w(0.8, t, FINGER_CLOSED, "lift the knife", rot))


BYPASS = os.environ.get("BYPASS", "")  # offline bypass tests only: "nofork", "press", "grab"


def _grab():
    """Bypass test: the left hand pinches the sausage itself instead of using the fork."""
    rot = grip_rotation(-math.pi / 2)  # fingers closing along x, across the link
    at = np.array([SX, PIN_Y, PF + R + 0.002])
    out = _lead("left", at + [0, 0, 0.06], rot, "over the sausage", FINGER_OPEN)
    out += [_w(1.2, at, FINGER_OPEN, "down round it", rot), _w(0.8, at, FINGER_CLOSED, "pinch it", rot)]
    return out


def demo():
    left, pin_tcp, r_tilt, d = _left()
    right = _right()
    if BYPASS in ("nofork", "grab"):
        out = {"right": right}
        for k in CUTS:
            _saw(right, Y_R + k * SL, KNIFE_TILT)
        if BYPASS == "grab":
            g = _grab()
            g.append(_w(max(sum(w.duration for w in right) - sum(w.duration for w in g), 0.5), g[-1].pos, FINGER_CLOSED,
                        "hold it", g[-1].yaw))
            out["left"] = g
        return out
    t_left = sum(w.duration for w in left)
    t_right = sum(w.duration for w in right)
    if t_right < t_left + 0.3:
        right.append(_w(t_left + 0.3 - t_right, right[-1].pos, FINGER_CLOSED, "wait for the fork", right[-1].yaw))
    for k in CUTS:
        _saw(right, Y_R + k * SL, KNIFE_TILT)
    # the fork comes out: levered over (handle down) about its tips until the tines tear out, then lifted clear
    t_cut = sum(w.duration for w in right)
    t_left = sum(w.duration for w in left)
    left.append(_w(max(t_cut - t_left, 0.3), pin_tcp, FINGER_CLOSED, "keep pinning", r_tilt))
    pin_tip = np.array([SX, PIN_Y, TOPZ]) + FORK_FIX + d * 0.008
    lev = math.radians(LEVER_DEG)
    r_lev = _rot(A_AX, -(HAND_TILT - lev)) @ G_FORK
    t_lev = pin_tip + _fork_grip(FORK_TILT - lev) - _fork_tines(FORK_TILT - lev) * 0.008
    left.append(_w(1.6, t_lev, FINGER_CLOSED, "lever the fork over until the tines come out", r_lev))
    left.append(_w(0.8, t_lev + [0, 0.02, 0.05], FINGER_CLOSED, "lift it clear", r_lev))
    # lay both down on the plate, clear of the food: the fork across the far side, the knife along the near rim
    r_lay = _rot(A_AX, -(PICK_PITCH - _fork_pitch)) @ G_FORK  # the fork level in the hand
    lay_f = np.array([PX + LAY_F_DX, -0.030 + FORK_GRIP_X, PF + 0.022])  # tips 3 cm right of the middle, far side
    left += [_w(1.5, lay_f + [0, 0, 0.03], FINGER_CLOSED, "fork over the plate's far side", r_lay),
             _w(0.8, lay_f, FINGER_CLOSED, "lower it", r_lay),
             _w(0.6, lay_f, FINGER_OPEN, "let go", r_lay),
             _w(0.5, lay_f + [0, 0, 0.03], FINGER_OPEN, "up", r_lay),
             _w(1.0, np.array([0.12, 0.24, TOP + 0.17]), FINGER_OPEN, "back out of the way", DOWN)]
    rk = _rz(math.pi / 2) @ DOWN  # the knife turned to point +y (to the left), along the near rim
    kyw = np.array([0.0, 1.0, 0.0])
    lay_k = np.array([PX - LAY_K_DX, PY + 0.0, PF + 0.008])  # the blade's middle
    tk = lay_k - (BLADE_L / 2) * kyw - KNIFE_GRIP * kyw + np.array([0, 0, HANDLE_Z + 0.003])
    right += [_w(1.6, tk + [0, 0, 0.04], FINGER_CLOSED, "knife turned along the plate's near rim", rk),
              _w(0.8, tk, FINGER_CLOSED, "lower it", rk),
              _w(0.6, tk, FINGER_OPEN, "let go", rk),
              _w(0.5, tk + [0, 0, 0.03], FINGER_OPEN, "up", rk),
              _w(1.0, np.array([0.12, -0.24, TOP + 0.17]), FINGER_OPEN, "back out of the way", DOWN)]
    t_l, t_r = sum(w.duration for w in left), sum(w.duration for w in right)
    (left if t_l < t_r else right).append(_w(abs(t_l - t_r) + 2.0, (left if t_l < t_r else right)[-1].pos, FINGER_OPEN,
                                             "wait", DOWN))
    (right if t_l < t_r else left).append(_w(2.0, (right if t_l < t_r else left)[-1].pos, FINGER_OPEN, "wait", DOWN))
    return {"left": left, "right": right}
