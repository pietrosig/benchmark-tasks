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
