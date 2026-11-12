"""Offline checks of the task file (no Isaac Sim): it loads, its scoring rules, its geometry and its demo's reach.

    YAM_ARENA=~/yam-isaac-lab-arena python -m pytest test_task.py
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, os.path.expanduser(os.environ.get("YAM_ARENA", "~/yam-isaac-lab-arena")))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from yam_arena import ik, station  # noqa: E402
from yam_arena.constants import CONTROL_HZ  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import load  # noqa: E402

TASK = load(HERE / "task.py")
import task as T  # noqa: E402  (the same module, for its constants)
from tankkit import FixtureState  # noqa: E402

RUN_LIMIT = 1200
TO_PLATFORM = (T.PX - T.TX, T.PY - T.TY)  # how far mx, my move the tank from its start to the platform


def rig(air=0.0, jet=0.0, mx=0.0, my=0.0, mz=0.0, shot=False, speed=0.0):
    p = np.array(T.RIG.pos, dtype=float)
    q = {"air": air, "jet": jet, "mx": mx, "my": my, "mz": mz, "button": 0.0}
    sp = {k: speed for k in q}
    broken = {k: False for k in q}
    broken["jet"] = shot
    return {"rig": FixtureState(pos=p, quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                                low=p, high=p, q=q, speed=sp, broken=broken),
            "platform": None}


def test_the_task_file_loads():
    assert TASK.scene == "station"
    assert [o.name for o in TASK.objects] == ["rig", "platform"]
    assert all(callable(f) for f in (TASK.success, TASK.distance, TASK.score, TASK.demo))


def test_still_connected_scores_zero():
    assert TASK.score(rig()) == 0.0


def test_disconnected_safely_but_not_moved_scores_half():
    assert TASK.score(rig(air=0.9)) == 0.5


def test_on_the_platform_scores_one():
    assert TASK.score(rig(air=1.0, mx=TO_PLATFORM[0], my=TO_PLATFORM[1], mz=T.PH)) == 1.0


@pytest.mark.parametrize("state", [dict(), dict(air=0.9), dict(air=1.0, mx=TO_PLATFORM[0], my=TO_PLATFORM[1], mz=T.PH)])
def test_a_tank_that_shot_away_scores_zero_whatever_happens_after(state):
    assert TASK.score(rig(shot=True, **state)) == 0.0


def test_the_platform_needs_the_tank_on_it_and_near_its_centre():
    off = T.ON_PLATFORM + 0.005
    assert TASK.score(rig(air=1.0, mx=TO_PLATFORM[0] + off, my=TO_PLATFORM[1], mz=T.PH)) == 0.5
    assert TASK.score(rig(air=1.0, mx=TO_PLATFORM[0], my=TO_PLATFORM[1], mz=0.0)) == 0.5  # beside it, on the table


def test_the_jet_is_counted_in_where_the_tank_is():
    xy = TASK.score.__globals__["tank_xy"](rig(jet=-0.1, mx=0.05))
    assert xy[0] == pytest.approx(T.TX - 0.05)


def test_success_needs_the_platform_and_a_still_tank():
    on = dict(air=1.0, mx=TO_PLATFORM[0], my=TO_PLATFORM[1], mz=T.PH)
    assert TASK.success(rig(**on)) is True
    assert TASK.success(rig(speed=0.05, **on)) is False
    assert TASK.success(rig(air=1.0)) is False


def test_distance_is_to_the_platform():
    assert TASK.distance(rig()) == pytest.approx(math.hypot(*TO_PLATFORM))
    assert TASK.distance(rig(mx=TO_PLATFORM[0], my=TO_PLATFORM[1])) == pytest.approx(0.0)


def test_the_jet_pushes_a_few_newtons_and_can_drive_the_tank_well_past_the_shot_speed():
    assert T.JET.stiffness * 10.0 == pytest.approx(T.JET_N) and 2.0 <= T.JET_N <= 4.0
    assert T.JET_N / T.JET.damping > T.SHOT  # unheld, it would reach this speed against the jet joint's damping alone
    assert T.JET.slam_speed == T.SHOT and T.JET.follows == "air"


def test_everything_stays_locked_until_the_release_is_pressed():
    for part in (T.AIR, T.JET, T.MX, T.MY, T.MZ):
        assert part.lock_until == T.RELEASED


def test_the_layout_fits_the_table_and_the_pipe_reaches_the_coupler():
    assert T.TL / 2 + T.PIPE_L == pytest.approx(T.COUPLER_X)
    assert T.TX + T.TL / 2 < T.WX - 0.06  # the tank stops short of the outlet's stand-off
    assert abs(T.PY - T.TY) > 0.12  # the platform is clear of the tank's start
    assert T.BUTTON_X > 0.15  # the coupler's button is far from the tank's handle: two hands are needed to hold and press


@pytest.mark.parametrize("arm", ["right", "left"])
def test_the_demo_is_reachable_within_the_joint_limits(arm):
    plan = ArmPlan(TASK.demo()[arm], station.BASES[arm])
    actions = np.array([plan.action7((k + 1) / CONTROL_HZ)[0] for k in range(round(plan.duration * CONTROL_HZ) + 5)])
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, (plan.worst_ik_mm, plan.worst_ik_deg)
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)


@pytest.mark.parametrize("arm", ["right", "left"])
def test_the_demo_never_jumps(arm):
    pts = np.array([w.pos for w in TASK.demo()[arm]][1:])
    assert np.linalg.norm(np.diff(pts, axis=0), axis=1).max() < 0.005  # under 10 cm/s


def test_the_demo_holds_the_tank_until_after_the_release_and_fits_the_run_limit():
    demo = TASK.demo()

    def when(label):
        for arm in demo.values():
            for i, w in enumerate(arm):
                if w.label == label:
                    return sum(x.duration for x in arm[:i])
        raise AssertionError(label)

    assert when("hold the tank") < when("press the release")
    assert max(sum(w.duration for w in arm) for arm in demo.values()) * 20 < 0.75 * RUN_LIMIT


def test_the_tank_can_slide_as_far_as_the_platform():
    assert T.MX.low <= TO_PLATFORM[0] <= T.MX.high
    assert T.MY.low <= TO_PLATFORM[1] <= T.MY.high
