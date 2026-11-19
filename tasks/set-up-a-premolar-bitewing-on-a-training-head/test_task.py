"""Offline checks (no simulator): the scoring rules on made-up states, and the scripted demo's reach.

    PYTHONPATH=~/yam-isaac-lab-arena uv run --python 3.12 --with numpy --with pytest python -m pytest test_task.py
"""

from __future__ import annotations

import importlib
import math
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


@pytest.fixture()
def task():
    for m in ("task", "head"):
        sys.modules.pop(m, None)
    return importlib.import_module("task")


FAR = {"left": {"tcp": np.array([0.0, 0.3, 1.1]), "open": 1.0}, "right": {"tcp": np.array([0.0, -0.3, 1.1]), "open": 1.0}}


def _quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        return np.array([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    r = math.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k]))
    v = [0.0, 0.0, 0.0]
    v[i] = r / 2
    v[j] = (R[j, i] + R[i, j]) / (2 * r)
    v[k] = (R[k, i] + R[i, k]) / (2 * r)
    return np.array([*v, (R[k, j] - R[j, k]) / (2 * r)])


def state(task, *, q, c, R, hands=None, moving=False):
    body = R @ task.R_START.T
    holder = SimpleNamespace(pos=np.asarray(c) + np.array([0, 0, task.TOP]), quat_xyzw=_quat(body),
                             lin_vel=np.zeros(3), ang_vel=np.zeros(3))
    head = SimpleNamespace(q={"jaw": q}, hands=hands or FAR)
    return {"head": head, "holder": holder}


def run(task, s, n=25):
    out = 0.0
    for _ in range(n):
        out = task.score(s)
    return out


def at_target(task, q, **kw):
    c, R = task.placed_pose(q, **kw)
    return state(task, q=q, c=c, R=R)


QB = -0.040


def test_start_scores_zero(task):
    s = state(task, q=-task.H.OPEN, c=task.P_START, R=task.R_START)
    assert run(task, s) == 0.0


def test_placed_and_bitten_scores_one(task):
    s = at_target(task, QB)
    e = task.errors(s)
    assert abs(e["u"]) < 1e-6 and abs(e["v"]) < 1e-6 and abs(e["d"] - task.H.D_TARGET) < 1e-6 and e["h"] < 0.01 and e["t"] < 0.01
    assert run(task, s) == 1.0


def test_must_be_still_for_a_second(task):
    s = at_target(task, QB)
    assert run(task, s, n=15) == 0.5  # not yet still for 20 steps: full position but not settled
    assert run(task, s, n=10) == 1.0


def test_jaw_open_scores_half(task):
    assert run(task, at_target(task, -task.H.OPEN)) == 0.5


def test_held_scores_half(task):
    s = at_target(task, QB)
    p, r = task.holder_tool(*task.placed_pose(QB))
    s["head"].hands = {"left": {"tcp": p + np.array([0, 0, task.TOP]), "open": 0.0}, "right": FAR["right"]}
    assert not task.released(s)
    assert run(task, s) == 0.5


def test_loose_band_scores_three_quarters(task):
    assert run(task, at_target(task, QB, fwd=0.006)) == 0.75  # 6 mm forward along the row
    assert run(task, at_target(task, QB, out=-0.007)) == 0.75  # 14 mm lingual of the teeth


def test_off_target_on_premolars_scores_quarter(task):
    assert run(task, at_target(task, QB, fwd=-0.010)) == 0.25


def test_far_off_scores_zero(task):
    assert run(task, at_target(task, QB, fwd=0.030)) == 0.0
    assert run(task, at_target(task, QB, lift=0.02)) == 0.0


def test_angulation(task):
    c, R = task.placed_pose(QB)
    a = math.radians(15.0)
    k = task.V_T  # turned about the head's vertical (the biting plane's normal)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    Rk = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
    s = state(task, q=QB, c=c, R=task.H.jaw_rot(QB) @ Rk @ task.R_T)
    assert abs(task.errors(s)["h"] - 15.0) < 0.5
    assert run(task, s) == 0.75


def test_fingers_in_the_mouth_latch_zero(task):
    s = at_target(task, QB)
    inside = task.H.to_world(np.array([task.H.XI + 0.01, 0.0, task.H.ZO])) + np.array([0, 0, task.TOP])
    s["head"].hands = {"left": {"tcp": inside, "open": 1.0}, "right": FAR["right"]}
    assert task.score(s) == 0.0
    s["head"].hands = FAR
    assert run(task, s) == 0.0  # for the rest of the run


@pytest.mark.parametrize("seed", ["0", "1", "2"])
def test_demo_reach(seed, monkeypatch):
    monkeypatch.setenv("TASK_SEED", seed)
    sys.modules.pop("task", None)
    t = importlib.import_module("task")
    from yam_arena.constants import LEFT_BASE_POS, RIGHT_BASE_POS
    from yam_arena.demo import ArmPlan

    plans = t.demo()
    for side, base in (("left", LEFT_BASE_POS), ("right", RIGHT_BASE_POS)):
        plan = ArmPlan(plans[side], base)
        tt = 0.0
        while tt < plan.duration:
            plan.action7(tt)
            tt += 0.05
        assert plan.worst_ik_mm < 3.0 and plan.worst_ik_deg < 2.0, side
        assert plan.duration < 60.0


def test_moving_holder_is_not_settled(task):
    s = at_target(task, QB)
    assert run(task, s) == 1.0
    s["holder"].pos = s["holder"].pos + np.array([0.002, 0.0, 0.0])  # it moved 2 mm this step
    assert task.score(s) == 0.5
    assert run(task, s, n=19) == 0.5
    assert task.score(s) == 1.0  # still again for 20 steps


def _local(axis, deg):
    """A turn about the holder's own u (0), v (1) or n (2) axis."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def turned(task, axis, deg, q=QB, **kw):
    c, R = task.placed_pose(q, **kw)
    return state(task, q=q, c=c, R=R @ _local(axis, deg))


@pytest.mark.parametrize("deg,expect", [(5, 1.0), (15, 0.75), (25, 0.0), (90, 0.0), (-90, 0.0), (180, 0.0)])
def test_in_plane_rotation(task, deg, expect):
    """Turned within its own plane about the plate's centre: the tab ends up up, down or at the back for 90 and 180."""
    s = turned(task, 2, deg)
    assert abs(task.errors(s)["r"] - abs(deg)) < 0.5
    assert run(task, s) == expect


@pytest.mark.parametrize("axis,deg", [(1, 180), (0, 180), (1, 90), (0, 90)])
def test_flipped_or_turned_holder_scores_zero(task, axis, deg):
    """Face turned away from the teeth (about v), upside down (about u), or turned a quarter turn: no credit."""
    assert run(task, turned(task, axis, deg)) == 0.0


def test_wrong_orientation_gets_no_quarter_credit(task):
    """The 0.25 tier (tab on the premolars, off position) also needs the right orientation."""
    assert run(task, at_target(task, QB, fwd=-0.010)) == 0.25
    for axis, deg in ((2, 90), (2, 180), (1, 180), (0, 180), (2, 30)):
        assert run(task, turned(task, axis, deg, fwd=-0.010)) == 0.0


def test_lingual_side_only(task):
    """The plate on the cheek side of the teeth (buccal) gets nothing."""
    assert run(task, at_target(task, QB, out=0.020)) == 0.0


def test_ring_is_fixed_on_the_holder(task):
    """In the simulation the ring is part of the holder's rigid body, 75 mm out on the beam axis: the score reads the
    plate itself, and the ring's place on the arm cannot change."""
    rings = [it for it in task.H.holder_items_and_start(task.TRAY_C, task.R_START)]
    assert task.H.RING[2] <= 75.0 <= task.H.RING[3]
    assert len(rings) > len(task.H.HOLDER_BOXES)  # the ring's segments are shapes of the same body


def _pose_with(task, du, dv, dout, h, t, r, q=QB):
    """The holder moved du/dv/dout (m) from the target and turned h (about the head's vertical), t (about the row)
    and r (in its plane), degrees."""
    c, R = task.placed_pose(q, lift=dv, fwd=du, out=dout)
    Rn = _local(1, h) @ _local(0, -t) @ _local(2, r)  # applied in the holder's own frame
    return state(task, q=q, c=c, R=R @ Rn)


def test_ring_worksheet_matches_plate_scoring(task):
    """The real-trial worksheet (bench plan angle and level-app tilt change, converted to the head's axes, then the
    1.3 mm per degree correction of the ring's offset) against the simulator's plate-based score, over random poses.
    Measured bounds (2,000 poses per band): full band u 0.8, v 0.7, d 0.1 mm, h 0.7, t 0.6 deg; loose band u 2.8,
    v 1.8, d 0.4 mm, h 2.0, t 1.4 deg. The test asserts slightly wider limits."""
    rng = np.random.default_rng(1)
    bands = (("full", (0.004, 0.003, 0.005, 10.0), 0.0010, 0.8), ("loose", (0.008, 0.006, 0.008, 20.0), 0.0030, 2.2))
    for band, lim, tol_m, tol_deg in bands:
        for _ in range(300):
            du, dv, dout = (rng.uniform(-x, x) for x in lim[:3])
            h, t, r = (rng.uniform(-lim[3], lim[3]) for _ in range(3))
            s = _pose_with(task, du, dv, dout, h, t, r)
            e, w = task.errors(s), task.ring_reading(s)
            for k in ("u", "v", "d"):
                assert abs(e[k] - w[k]) <= tol_m, (band, k, e[k], w[k], h, t)
            assert abs(abs(w["h"]) - e["h"]) <= tol_deg and abs(abs(w["t"]) - e["t"]) <= tol_deg, (band, w, e)


def test_a_turn_without_the_correction_would_misread(task):
    """Why the corrections are needed: a full-score placement turned 6 degrees moves the ring 7.8 mm along the row, and
    the bench angles differ from the head's (the head is reclined): a pure head turn of 10 degrees reads as about 9 deg
    in plan and 4 deg of tilt, which the conversion maps back."""
    s = _pose_with(task, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0)
    w = task.ring_reading(s)
    assert abs(w["u"] + 0.075 * math.sin(math.radians(w["h"])) - 0.075 * math.sin(math.radians(6.0))) < 0.0008
    assert abs(w["u"]) < 0.0008 and run(task, s) == 1.0
    w = task.ring_reading(_pose_with(task, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0))
    assert 8.0 < w["plan"] < 11.0 and 2.0 < w["tilt"] < 5.0 and abs(w["h"] - 10.0) < 0.8 and abs(w["t"]) < 0.8
