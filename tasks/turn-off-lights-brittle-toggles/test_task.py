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
from togglekit import FixtureState  # noqa: E402

RUN_LIMIT = 1200
ON, OFF = T.THROW, -T.THROW


def panel(toggles=(ON, ON, OFF, ON), snapped=(), speed=0.0):
    p = np.array(T.PANEL.pos, dtype=float)
    q = {f"toggle{i}": a for i, a in enumerate(toggles)}
    q.update({f"lever{i}": 0.0 for i in range(4)})
    broken = {f"toggle{i}": False for i in range(4)}
    broken.update({f"lever{i}": i in snapped for i in range(4)})
    return {"panel": FixtureState(pos=p, quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                                  low=p, high=p, q=q, speed={f"toggle{i}": speed for i in range(4)}, broken=broken)}


def test_the_task_file_loads():
    assert TASK.scene == "station"
    assert [o.name for o in TASK.objects] == ["panel"]
    assert all(callable(f) for f in (TASK.success, TASK.distance, TASK.score, TASK.demo))


def test_three_lamps_are_lit_at_the_start_and_one_is_off():
    assert T.LIT_AT_START == (0, 1, 3) and T.OFF_AT_START == (2,)
    assert TASK.score(panel()) == 0.0
    assert TASK.distance(panel()) == 3.0


@pytest.mark.parametrize("toggles, expected", [((ON, ON, OFF, ON), 0.0), ((OFF, ON, OFF, ON), 1 / 3),
                                               ((OFF, OFF, OFF, ON), 2 / 3), ((OFF, OFF, OFF, OFF), 1.0),
                                               ((ON, ON, ON, ON), 0.0), ((OFF, OFF, ON, OFF), 1.0)])
def test_a_third_for_each_lamp_turned_off(toggles, expected):
    assert TASK.score(panel(toggles)) == pytest.approx(expected)


def test_a_toggle_counts_as_off_only_past_12_degrees():
    assert TASK.score(panel((math.radians(-11), OFF, OFF, OFF))) == pytest.approx(2 / 3)
    assert TASK.score(panel((math.radians(-13), OFF, OFF, OFF))) == 1.0


@pytest.mark.parametrize("which", [0, 1, 2, 3])
def test_any_snapped_lever_scores_zero_even_the_one_already_off(which):
    assert TASK.score(panel((OFF, OFF, OFF, OFF), snapped=(which,))) == 0.0


def test_success_needs_all_off_nothing_snapped_and_still_toggles():
    assert TASK.success(panel((OFF, OFF, OFF, OFF))) is True
    assert TASK.success(panel((OFF, OFF, OFF, OFF), speed=0.5)) is False
    assert TASK.success(panel((OFF, OFF, OFF, ON))) is False
    assert TASK.success(panel((OFF, OFF, OFF, OFF), snapped=(2,))) is False


def test_distance_counts_lit_lamps():
    assert TASK.distance(panel((OFF, ON, OFF, ON))) == 2.0
    assert TASK.distance(panel((OFF, OFF, OFF, OFF))) == 0.0


def test_a_flip_needs_far_less_force_than_a_snap():
    push = T.LEVER - 0.006  # the hand pushes just under the knob: this far above the lever's base
    flip_n = T.TOGGLE_K * (T.THROW - T.CLICK) / (push + 0.004)  # the most it takes, just before the click
    snap_n = T.BEND_K * T.SNAP_AT / push
    assert flip_n < 0.25 and 3.0 < snap_n < 4.0
    assert snap_n > 10 * flip_n


def test_the_click_point_is_just_past_the_middle_and_the_demo_pulls_past_it():
    assert -0.1 < T.CLICK < 0 and T.PULL_TO < T.CLICK
    for part in T.PARTS:
        if part.name.startswith("toggle") and part.retarget:
            assert part.retarget == (part.name, T.CLICK, OFF)


def test_the_lamps_and_toggles_sit_on_the_board_and_do_not_touch():
    half_x, half_y = T.BOARD[0] / 2, T.BOARD[1] / 2
    for y in T.YS:
        assert abs(y) + 0.011 < half_y
    assert abs(T.LAMP_X) + 0.008 < half_x and T.PIVOT_X + 0.015 < half_x
    for i in range(4):  # off, the knob stays clear of its lamp
        assert T.tip(i, OFF)[0] - T.KNOB_D / 2 > T.X + T.LAMP_X + 0.009
    assert min(abs(a - b) for a in T.YS for b in T.YS if a != b) > 0.06


@pytest.mark.parametrize("arm", ["right", "left"])
def test_the_demo_is_reachable_within_the_joint_limits(arm):
    plan = ArmPlan(TASK.demo()[arm], station.BASES[arm])
    actions = np.array([plan.action7((k + 1) / CONTROL_HZ)[0] for k in range(round(plan.duration * CONTROL_HZ) + 5)])
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, (plan.worst_ik_mm, plan.worst_ik_deg)
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)


@pytest.mark.parametrize("arm", ["right", "left"])
def test_the_demo_is_smooth(arm):
    w = TASK.demo()[arm][1:]  # after the first move from ready
    pts = np.array([x.pos for x in w])
    dt = np.array([x.duration for x in w])
    v = np.linalg.norm(np.diff(pts, axis=0), axis=1) / dt[1:]
    assert v.max() <= T.V_MOVE + 1e-6  # never faster than 2.5 cm/s
    assert (np.abs(np.diff(v)) / dt[2:]).max() < 0.06  # accelerates gently


def test_the_demo_pulls_each_lit_lever_past_its_click_point_and_leaves_the_off_one_alone():
    demo = TASK.demo()
    pts = np.array([w.pos for arm in demo.values() for w in arm])
    for i in range(4):
        near = pts[np.abs(pts[:, 1] - T.YS[i]) < 0.005]
        low = near[near[:, 2] < T.tip(i)[2]]
        if i in T.LIT_AT_START:
            assert low[:, 0].min() < T.tip(i, T.CLICK)[0] - 0.001  # past the click point (the path's corner ends it just short of PULL_TO)
        else:
            assert len(low) == 0


def test_the_demo_fits_the_run_limit():
    assert max(sum(w.duration for w in arm) for arm in TASK.demo().values()) * 20 < RUN_LIMIT / 2
