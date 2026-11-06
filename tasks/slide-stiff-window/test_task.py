"""Offline checks of the task file (no Isaac Sim): it loads, its scoring rules, its geometry and its demo's reach.

    YAM_ARENA=~/yam-isaac-lab-arena python -m pytest test_task.py
"""

from __future__ import annotations

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
from slidekit import FixtureState  # noqa: E402

RUN_LIMIT = 1200


def window(opened=0.0, give=0.0, torn=False, speed=0.0):
    p = np.array(T.WINDOW.pos, dtype=float)
    return {"window": FixtureState(pos=p, quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                                   low=p, high=p, q={"sash": -opened, "handle": -give},
                                   speed={"sash": -speed, "handle": 0.0}, broken={"sash": False, "handle": torn})}


def test_the_task_file_loads():
    assert TASK.scene == "station"
    assert [o.name for o in TASK.objects] == ["window"]
    assert all(callable(f) for f in (TASK.success, TASK.distance, TASK.score, TASK.demo))


@pytest.mark.parametrize("opened, expected", [(0.0, 0.0), (0.039, 0.0), (0.04, 0.25), (0.079, 0.25), (0.08, 0.5),
                                              (0.119, 0.5), (0.12, 0.75), (0.149, 0.75), (0.15, 1.0), (0.24, 1.0)])
def test_the_score_follows_how_far_the_sash_is_open(opened, expected):
    assert TASK.score(window(opened)) == expected


def test_doing_nothing_scores_zero():
    assert TASK.score(window()) == 0.0


@pytest.mark.parametrize("opened", [0.0, 0.06, 0.15, 0.24])
def test_a_torn_handle_scores_zero(opened):
    assert TASK.score(window(opened, give=0.14, torn=True)) == 0.0


def test_a_handle_under_strain_but_on_still_counts():
    assert TASK.score(window(0.18, give=0.015)) == 1.0


def test_success_needs_the_goal_the_handle_and_a_still_sash():
    assert TASK.success(window(0.18)) is True
    assert TASK.success(window(0.18, speed=0.05)) is False
    assert TASK.success(window(0.14)) is False
    assert TASK.success(window(0.18, torn=True)) is False


def test_distance_counts_down_to_the_goal():
    assert TASK.distance(window(0.0)) == pytest.approx(T.GOAL)
    assert TASK.distance(window(0.10)) == pytest.approx(T.GOAL - 0.10)
    assert TASK.distance(window(0.20)) == 0.0


def test_the_tear_force_is_about_14_newtons():
    assert T.HANDLE_K * T.TEAR_AT == pytest.approx(14.4)
    assert T.HANDLE.breaks_at == T.TEAR_AT and T.HANDLE.low < -T.TEAR_AT  # it can tear before it hits its stop


def test_a_slow_pull_is_well_under_the_tear_force_and_a_fast_one_over_it():
    slow = T.SASH_DAMPING * (T.OPEN + 0.012) / T.PULL_S  # the demo's pull, about 2 cm/s
    fast = T.SASH_DAMPING * 0.10
    assert slow < 0.5 * T.HANDLE_K * T.TEAR_AT < T.HANDLE_K * T.TEAR_AT < fast


def test_the_closed_sash_fills_the_left_half_and_can_open_past_the_goal():
    assert T.SASH_Y + T.SASH_W / 2 == pytest.approx(T.Y_L)
    assert T.SASH_Y - T.SASH_W / 2 == pytest.approx(T.MEET_Y)
    assert T.SASH_TRAVEL > T.OPEN > T.GOAL
    assert T.MEET_Y - T.SASH_TRAVEL > T.Y_R  # fully open, it still stops short of the right jamb


def test_the_sash_slides_in_front_of_the_fixed_pane():
    sash_back = T.SASH_FRONT + T.SASH_D
    pane_front = T.PANE_X - 0.006
    assert sash_back < pane_front


def test_the_handle_stands_clear_of_the_sash_and_frame():
    bar_front = T.X + T.SASH_FRONT + T.SASH_D / 2 + T._FACE - T.STAND - T.BAR_X
    assert bar_front < T.X - T.DEPTH / 2 - 0.02  # the bar stands 2 cm or more proud of the frame's face


@pytest.mark.parametrize("arm", ["right"])
def test_the_demo_is_reachable_within_the_joint_limits(arm):
    plan = ArmPlan(TASK.demo()[arm], station.BASES[arm])
    actions = np.array([plan.action7((k + 1) / CONTROL_HZ)[0] for k in range(round(plan.duration * CONTROL_HZ) + 5)])
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, (plan.worst_ik_mm, plan.worst_ik_deg)
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)


def test_the_demo_pulls_the_handle_past_the_goal():
    pts = np.array([w.pos for w in TASK.demo()["right"]])
    start_y = T.SASH_Y + T.HANDLE_Y
    assert start_y - pts[:, 1].min() > T.GOAL + 0.03


def test_the_demo_never_jumps():
    pts = np.array([w.pos for w in TASK.demo()["right"]][1:])  # after the first move from ready
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert steps.max() < 0.01  # under 1 cm per 0.05 s waypoint: under 20 cm/s


def test_the_demo_fits_the_run_limit():
    assert sum(w.duration for w in TASK.demo()["right"]) * 20 < RUN_LIMIT / 2
