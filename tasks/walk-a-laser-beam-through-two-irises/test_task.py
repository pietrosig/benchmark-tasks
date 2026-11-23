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


def test_far_spot_and_meter_dark_when_near_aperture_blocks():
    # every seeded start: the beam lands on aperture 1's plate off the hole, so no light reaches aperture 2 or the
    # photodiode: the far spot is hidden (pushed into its plate, centred), the meter reads 0, the near spot is drawn
    for s, off in enumerate(task.OFFSETS):
        q = {k: 2 * math.pi * t for k, t in zip(task.KNOB_NAMES, off)}
        b = task.beam(q)
        assert b["r1"] > task.R_PASS and b["t1"] == 0.0 and b["power"] == 0.0, s
        t = task.spot_targets(q)
        assert t["s2_x"] == task.SPOT_HIDE and t["s2_y"] == 0.0 and t["s2_z"] == 0.0, s
        assert t["s1_x"] == 0.0 and t["bar"] == 0.0, s


def test_clipping_is_consistent_between_meter_and_spots():
    # sweep one knob through and past alignment: the far spot is drawn exactly when light passes aperture 1, the meter
    # reads the product of the two apertures' transmissions, and is 0 whenever the far spot is hidden
    seen = set()
    for t in np.linspace(-0.6, 0.6, 241):
        q = {**{k: 0.0 for k in task.KNOB_NAMES}, "m1_up": 2 * math.pi * t}
        b = task.beam(q)
        tg = task.spot_targets(q)
        passes = b["r1"] <= task.R_PASS
        seen.add(passes)
        assert (tg["s2_x"] == 0.0) == passes
        if passes:
            assert b["power"] == pytest.approx(math.exp(-2 * (b["r1"] ** 2 + b["r2"] ** 2) / task.W_EFF ** 2))
        else:
            assert b["power"] == 0.0 and tg["bar"] == 0.0
    assert seen == {True, False}


def test_knob_ticks_do_not_mark_the_aligned_setting():
    # the drawn tick on each knob sits at a random angle per seed and knob: it does not mark the aligned setting
    # (aligned = joint angle 0, where the tick shows at its drawn phase) in any consistent way across seeds
    import importlib
    phases = []
    for seed in range(20):
        os.environ["TASK_SEED"] = str(seed)
        phases.append(importlib.reload(task).MARK_PHASE)
    os.environ.pop("TASK_SEED")
    importlib.reload(task)
    for k in task.KNOB_NAMES:
        a = np.array([p[k] for p in phases])
        assert np.ptp(np.mod(a, 2 * math.pi)) > math.pi, k  # spread round the knob
        assert abs(np.mean(np.exp(1j * a))) < 0.6, k  # no preferred angle


def test_display_parts_start_at_their_targets():
    # the first frame already shows the start state: far spot hidden, near spot off the hole, meter at 0
    import beamkit
    want = task.spot_targets({k: 2 * math.pi * t for k, t in task.START_TURNS.items()})
    for part, v in want.items():
        assert beamkit._START[("bench", part)] == pytest.approx(v), part
    assert want["s2_x"] == task.SPOT_HIDE and want["s1_x"] == 0.0 and want["bar"] == 0.0
    assert math.hypot(want["s1_y"], want["s1_z"]) > task.R_PASS
