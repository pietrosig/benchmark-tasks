"""Offline checks (no simulator): the scoring rules on made-up states, the geometry of the holders, and the scripted
demo's reach.

    git clone https://github.com/kairosiann/yam-isaac-lab-arena ~/yam-isaac-lab-arena
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
    for m in ("task", "ebkit"):
        sys.modules.pop(m, None)
    return importlib.import_module("task")


def _xyzw(r):
    from yam_arena import ik

    return ik.matrix_to_quat_xyzw(r)


def state(task, *, axial=0.0, off=0.0, tilt_deg=0.0, lateral=0.0, roll_deg=0.0, plunger=0.0, vz=0.0, at_start=False, dz=0.0):
    """The battery at a pose given in the tube frame (bottom back corner ``axial`` up the tube, ``off`` off it, tilted
    ``tilt_deg`` about its bottom end), or where it starts on the bench."""
    if at_start:
        p, r = task.START_POS, task.START_R
    else:
        c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
        rr = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        p, r = task._bat_world(np.array([lateral, axial, off]), task._rx(math.radians(tilt_deg)) @ rr)
    bat = SimpleNamespace(pos=np.asarray(p) + np.array([0, 0, task.TOP + dz]), quat_xyzw=_xyzw(r),
                          lin_vel=np.array([0.0, 0.0, vz]), ang_vel=np.zeros(3))
    hands = {"left": {"tcp": np.array([0.0, 0.3, 1.0]), "open": 1.0}, "right": {"tcp": np.array([0.0, -0.3, 1.0]), "open": 1.0}}
    bike = SimpleNamespace(q={"plunger": plunger, "bolt": 0.0}, hands=hands)
    return {"bike": bike, "battery": bat}


def test_start_scores_zero(task):
    assert task.score(state(task, at_start=True)) == 0.0


def test_start_pose_lies_on_the_bench(task):
    # lying on its flank: lowest point just above the bench, its back toward the arms
    corners = np.array([[x, y, z] for x in (-task.BW / 2, task.BW / 2) for y in (0, task.BL) for z in (0, task.BD)])
    w = task.START_POS + corners @ task.START_R.T
    assert 0.0 < w[:, 2].min() < 0.003
    assert np.isclose(w[:, 0].min(), task.START_POS[0])  # the back face is the face nearest the arms


def test_home_unlatched_is_half(task):
    # swung in, resting on the plunger (6 mm proud): 0.5; not yet clicked
    assert task.score(state(task, off=0.0, tilt_deg=math.degrees(math.atan(0.006 / task.BL)), plunger=0.0)) == 0.5


def test_bottom_in_pocket_tilted_is_quarter(task):
    assert task.score(state(task, tilt_deg=5.0)) == 0.25


def test_too_far_up_the_tube_is_zero(task):
    assert task.score(state(task, axial=0.02, tilt_deg=5.0)) == 0.0
    assert task.score(state(task, axial=0.004, tilt_deg=0.0)) == 0.0


def test_lifted_off_the_seat_is_zero(task):
    assert task.score(state(task, off=0.01, tilt_deg=3.0)) == 0.0


def test_twisted_is_zero(task):
    assert task.score(state(task, roll_deg=8.0)) == 0.0
    assert task.score(state(task, lateral=0.006)) == 0.0


def test_pressed_home_latches_for_good(task):
    s = state(task, plunger=-task.PLUNGER_TRAVEL)
    assert task.score(s) == 1.0
    assert task._latched[0]
    import ebkit

    assert ebkit._FLAGS.get("latched")
    # still home (the bolt holds it): stays 1.0
    assert task.score(state(task, plunger=-task.PLUNGER_TRAVEL + 0.0005)) == 1.0
    # pulled out after it latched (only the simulator could): no longer home, no credit
    assert task.score(state(task, axial=0.03, off=0.02, tilt_deg=10.0)) == 0.0


def test_success_needs_the_hands_off(task):
    s = state(task, plunger=-task.PLUNGER_TRAVEL)
    assert task.score(s) == 1.0 and task.success(s)  # hands far away
    p, r = task._bat_world(np.zeros(3), np.eye(3))
    s["bike"].hands["left"]["tcp"] = p + r @ np.array([-0.03, 0.26, task.BD / 2]) + np.array([0, 0, task.TOP])
    assert task.score(s) == 1.0 and not task.success(s)


def test_pressing_the_plunger_without_the_battery_home_does_not_latch(task):
    assert task.score(state(task, axial=0.02, plunger=-task.PLUNGER_TRAVEL)) == 0.0
    assert not task._latched[0]


def test_a_fall_scores_zero_for_good(task):
    assert task.score(state(task, at_start=True, dz=0.10)) == 0.0
    assert not task._dropped[0]
    assert task.score(state(task, at_start=True, dz=0.09)) == 0.0  # lowered 1 cm in a step: 0.2 m/s, not a fall
    assert not task._dropped[0]
    assert task.score(state(task, at_start=True, dz=0.05)) == 0.0  # 4 cm in a step: 0.8 m/s, a fall
    assert task._dropped[0]
    assert task.score(state(task, plunger=-task.PLUNGER_TRAVEL)) == 0.0


def test_holder_geometry(task):
    # the contact block clears the pocket's floor up to about 7 degrees of tilt, and fits under the lip
    lo_z, hi_z = task.TONGUE[2]
    tip = -task.TONGUE[1][0]
    assert 6.0 <= math.degrees(math.asin(lo_z / tip)) <= 8.0
    assert hi_z < task.LIP_LOW and task.LIP_LOW - hi_z <= 0.004
    assert task.TONGUE[0][1] < task.POCKET_SIDE <= task.TONGUE[0][1] + 0.004
    assert -task.TONGUE[1][0] < -task.POCKET_END <= -task.TONGUE[1][0] + 0.002
    # the bolt, run out, sits in the battery's notch without touching its end
    assert 0.3505 - task.BOLT_TRAVEL - 0.0075 > task.NOTCH_Y
    # the plunger's preload: 15 N up, 30 N pressed flush
    assert task.PLUNGER_K * task.PLUNGER_TRAVEL == pytest.approx(15.0)


def test_demo_is_reachable(task):
    from yam_arena.demo import ArmPlan

    plans = task.demo()
    for side, base in (("left", (0.0, 0.305, 0.75)), ("right", (0.0, -0.305, 0.75))):
        ap = ArmPlan(plans[side], base)
        t = 0.0
        while t < ap.duration:
            ap.action7(t)
            t += 0.05
        assert ap.worst_ik_mm < 5.0, (side, ap.worst_ik_mm)
        assert ap.worst_ik_deg < 6.0, (side, ap.worst_ik_deg)
        assert ap.duration * 20 < 1600 / 2
