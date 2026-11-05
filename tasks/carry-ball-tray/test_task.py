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
from yam_arena.constants import CONTROL_HZ, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402

TASK = load(HERE / "task.py")
import task as T  # noqa: E402  (the same module, for its constants)

PAD_HALF_WIDTH = 0.0201  # the finger collision hulls' half width within 15 mm of the TCP
RUN_LIMIT = 1200


def state(pos, quat=(0, 0, 0, 1.0), vel=(0, 0, 0)):
    pos = np.array(pos, dtype=float)
    return ObjectState(pos=pos, quat_xyzw=np.array(quat, dtype=float), lin_vel=np.array(vel, dtype=float),
                       ang_vel=np.zeros(3), low=pos, high=pos)


def scene(x=T.FIX_X, y=T.FIX_Y, floor=T.TOP + T.PLATE_H, in_tray=6, spilled=None, tilt=0.0, vel=(0, 0, 0)):
    """The tray with its underside at height ``floor``, ``in_tray`` balls on it and the rest at ``spilled``."""
    half = tilt / 2
    tray = state((x, y, floor + T.TRAY_BOTTOM_TO_CENTRE), (math.sin(half), 0, 0, math.cos(half)), vel)
    objects = {"tray": tray}
    ball_z = floor + T.FLOOR_T + T.BALL_R
    for i, ball in enumerate(T.BALLS):
        dx, dy = ball.pos[0] - T.START_X, ball.pos[1] - T.START_Y
        where = (x + dx, y + dy, ball_z) if i < in_tray else (spilled or (0.2, 0.3, T.TOP + T.BALL_R))
        objects[ball.name] = state(where)
    return objects


def test_the_task_file_loads():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {"tray", *T.BALL_NAMES}
    assert len(T.BALL_NAMES) == 6
    assert all(callable(f) for f in (TASK.success, TASK.distance, TASK.score, TASK.demo))


@pytest.mark.parametrize("in_tray, expected", [(6, 1.0), (5, 0.75), (4, 0.5), (3, 0.5), (2, 0.25), (1, 0.25), (0, 0.0)])
def test_the_score_counts_the_balls_with_the_tray_seated(in_tray, expected):
    assert TASK.score(scene(in_tray=in_tray)) == expected


def test_doing_nothing_scores_zero():
    assert TASK.score(scene(x=T.START_X, y=T.START_Y, floor=T.TOP + T.REST_GAP)) == 0.0


def test_a_tray_held_above_the_pocket_scores_zero():
    assert TASK.score(scene(floor=T.TOP + T.PLATE_H + 0.02)) == 0.0


def test_a_tray_caught_on_a_rail_scores_zero():
    # one side of the pan up on a rail, the other on the floor: its underside's centre is half a rail up
    tilt = math.asin(T.RAIL_H / T.TRAY_X)
    assert TASK.score(scene(floor=T.TOP + T.PLATE_H + T.RAIL_H / 2, tilt=tilt)) == 0.0


def test_a_tray_still_held_in_the_pocket_counts():
    # measured in Isaac: 1.4 mm up and 1 degree tilted while the arms still hold it
    assert TASK.score(scene(floor=T.TOP + T.PLATE_H + 0.0014, tilt=math.radians(1.0))) == 1.0


def test_a_tray_beside_the_pocket_scores_zero():
    assert TASK.score(scene(x=T.FIX_X - 0.02)) == 0.0


def test_a_ball_on_the_fixture_is_not_in_the_tray():
    on_plate = (T.FIX_X + T.TRAY_X / 2 + 0.002, T.FIX_Y, T.TOP + T.PLATE_H + T.BALL_R)
    assert TASK.score(scene(in_tray=5, spilled=on_plate)) == 0.75


def test_a_ball_held_above_the_tray_is_not_in_it():
    above = (T.FIX_X, T.FIX_Y, T.TOP + T.PLATE_H + T.FLOOR_T + T.BALL_R + 0.02)
    assert TASK.score(scene(in_tray=5, spilled=above)) == 0.75


def test_success_needs_everything_still():
    assert TASK.success(scene()) is True
    assert TASK.success(scene(vel=(0.1, 0, 0))) is False
    assert TASK.success(scene(in_tray=5)) is False


def test_the_balls_start_on_the_tray_inside_its_lip():
    objects = scene(x=T.START_X, y=T.START_Y, floor=T.TOP + T.REST_GAP)
    assert T.balls_in_tray(objects) == 6
    for ball in T.BALLS:
        assert abs(ball.pos[0] - T.START_X) + T.BALL_R < T.TRAY_X / 2 - T.LIP_T
        assert abs(ball.pos[1] - T.START_Y) + T.BALL_R < T.TRAY_Y / 2 - T.LIP_T


def test_the_balls_do_not_touch_each_other():
    gaps = [np.linalg.norm(np.array(a.pos) - np.array(b.pos)) for a in T.BALLS for b in T.BALLS if a is not b]
    assert min(gaps) > 2 * T.BALL_R


def test_the_lip_is_lower_than_a_balls_centre():
    assert T.LIP_H < T.BALL_R


def test_the_tray_fits_the_pocket_and_cannot_be_pushed_in():
    assert T.POCKET_X - T.TRAY_X == pytest.approx(2 * T.CLEAR)
    assert T.POCKET_Y - T.TRAY_Y == pytest.approx(2 * T.CLEAR)
    assert T.LUG_GAP > T.LUG_X
    assert T.PLATE_H > T.LUG_H / 2  # the plate's face is well above the pan's lip


def test_the_start_clears_the_fixture():
    assert T.START_X + T.TRAY_X / 2 < T.FIX_X - T.PLATE_X / 2


def test_the_pads_clear_the_fixture_and_the_table():
    assert T.GRIP_Y - PAD_HALF_WIDTH > T.PLATE_Y / 2
    assert T.GRIP_Y + PAD_HALF_WIDTH <= T.LUG_END + 0.0002
    assert T.GRIP_ABOVE_BOTTOM - PAD_TIP_PAST_TCP > 0


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_demo_is_reachable_within_the_joint_limits(arm):
    plan = ArmPlan(TASK.demo()[arm], station.BASES[arm])
    actions = np.array([plan.action7((k + 1) / CONTROL_HZ)[0] for k in range(round(plan.duration * CONTROL_HZ) + 5)])
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, (plan.worst_ik_mm, plan.worst_ik_deg)
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)


def test_both_arms_move_together():
    left, right = TASK.demo()["left"], TASK.demo()["right"]
    assert [w.duration for w in left] == [w.duration for w in right]
    for a, b in zip(left, right):
        assert a.pos[0] == b.pos[0] and a.pos[2] == b.pos[2] and a.pos[1] == -b.pos[1]


def test_the_demo_fits_the_run_limit():
    longest = max(sum(w.duration for w in waypoints) for waypoints in TASK.demo().values())
    assert longest * 20 < RUN_LIMIT
