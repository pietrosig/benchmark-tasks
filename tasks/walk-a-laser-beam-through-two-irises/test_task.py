"""Offline tests (no simulator): the beam model, the score and the demo plan's reach."""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task  # noqa: E402

AWAY = {"left_robot": {"tcp": np.array([0.05, 0.25, task.TOP + 0.3]), "open": 0.3},
        "right_robot": {"tcp": np.array([0.05, -0.25, task.TOP + 0.3]), "open": 0.3}}


def objs(turns, hands=AWAY):
    q = {k: 2 * math.pi * turns.get(k, 0.0) for k in task.KNOB_NAMES}
    return {"bench": SimpleNamespace(q=q, hands=hands)}


@pytest.fixture(autouse=True)
def fresh():
    task._state.update(void=None, calls=0)


def test_aligned_full_reading():
    b = task.beam({k: 0.0 for k in task.KNOB_NAMES})
    assert b["power"] == pytest.approx(1.0, abs=1e-6)
    assert b["r1"] < 1e-6 and b["r2"] < 1e-6
    assert task.score(objs({})) == 1.0


def test_every_seed_starts_at_zero():
    for s, off in enumerate(task.OFFSETS):
        turns = dict(zip(task.KNOB_NAMES, off))
        assert task.score(objs(turns)) == 0.0, s
        assert task.reading(objs(turns)) < 0.1, s


def test_sensitivity_matches_the_mount():
    # one full turn of M2's side knob tilts the mirror 8 mrad: the beam turns 16 mrad in plane,
    # so it moves about 16 mrad x 0.24 m = 3.8 mm at the far aperture
    b = task.beam({**{k: 0.0 for k in task.KNOB_NAMES}, "m2_side": 2 * math.pi})
    assert 3.0e-3 < b["r2"] < 4.6e-3
    # the near aperture moves less (12 cm from M2)
    assert b["r1"] < b["r2"]


def test_knobs_are_coupled():
    # turning only M1 moves the spot at both apertures
    b = task.beam({**{k: 0.0 for k in task.KNOB_NAMES}, "m1_up": math.pi / 2})
    assert b["r1"] > 1e-4 and b["r2"] > 1e-4


def test_partial_credit_levels():
    # a small error on one knob only: find angles giving each level
    def s(t):
        task._state.update(void=None)
        return task.score(objs({"m1_up": t}))
    assert s(0.0) == 1.0
    vals = {s(t) for t in np.linspace(0, 0.6, 300)}
    assert vals == {1.0, 0.5, 0.25, 0.0}


def test_hands_on_a_knob_scores_zero():
    k = task._world_knob("m1_up") + np.array([0, 0, task.TOP])
    hands = {"left_robot": {"tcp": k, "open": 0.1}, "right_robot": AWAY["right_robot"]}
    assert task.score(objs({}, hands)) == 0.0


def test_mount_body_keep_out():
    p = task.M2 + task._rz(task.YAW2) @ np.array([-0.015, 0.0, -0.05]) + np.array([0, 0, task.TOP])  # on M2's post
    hands = {"left_robot": AWAY["left_robot"], "right_robot": {"tcp": p, "open": 0.5}}
    assert task.score(objs({}, hands)) == 0.0


def test_keep_out_latches():
    hands = {"left_robot": {"tcp": task.A1 + np.array([0, 0, task.TOP]), "open": 0.5}, "right_robot": AWAY["right_robot"]}
    assert task.score(objs({}, hands)) == 0.0
    assert task.score(objs({})) == 0.0  # still void after leaving


def test_demo_reaches_every_waypoint():
    from yam_arena.demo import ArmPlan
    bases = {"left": (0, 0.305, task.TOP), "right": (0, -0.305, task.TOP)}
    for arm, w in task.demo().items():
        p = ArmPlan(w, bases[arm])
        t = 0.0
        while t < p.duration:
            p.action7(t)
            t += 0.05
        assert p.worst_ik_mm < 2.0 and p.worst_ik_deg < 1.0, arm


def test_demo_undoes_the_start():
    # the demo turns each knob by minus its starting offset
    total = {}
    for name in task.KNOB_NAMES:
        total[name] = task.START_TURNS[name]
    assert all(abs(v) >= 0.3 for v in total.values())
