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
from drawerkit import CabinetState, closed_bounds  # noqa: E402
from yam_arena import ik, station  # noqa: E402
from yam_arena.constants import CONTROL_HZ, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import ObjectState, load  # noqa: E402

TASK = load(HERE / "task.py")
PAD_HALF_WIDTH = 0.0201  # the finger collision hulls' half width within 15 mm of the TCP
import task as T  # noqa: E402  (the same module, for its constants)


def bar_state(lift=0.0, dx=0.0, slide=0.0):
    centre = np.array(T.BAR_POS) + [dx, slide, lift]
    half = np.array(T.BAR_SIZE) / 2
    return ObjectState(pos=centre, quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                       low=centre - half, high=centre + half)


def cabinet_state(opening=0.0, speed=0.0):
    low, high = closed_bounds(T.CABINET)
    names = [d.name for d in T.CABINET.drawers]
    return CabinetState(pos=np.array(T.CABINET.pos), quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3),
                        ang_vel=np.zeros(3), low=low, high=high,
                        opening={n: (opening if n == "top_drawer" else 0.0) for n in names},
                        speed={n: (speed if n == "top_drawer" else 0.0) for n in names})


def test_the_task_file_loads_with_a_fixed_cabinet_and_a_loose_bar():
    assert TASK.scene == "station" and TASK.names == ("cabinet", "bar")
    assert TASK.free_objects == ("bar",)
    assert TASK.success_hold_s == 0.5 and TASK.score_over == "end"
    assert all(callable(f) for f in (TASK.success, TASK.distance, TASK.score, TASK.demo))


@pytest.mark.parametrize(
    "opening, speed, slide, expected_score, expected_success",
    [
        (0.0, 0.0, 0.0, 0.0, False),  # nothing done
        (0.02, 0.0, 0.0, 0.0, False),  # pulled against the bar
        (0.0, 0.0, 0.16, 0.0, False),  # bar out of the right sleeve only
        (0.0, 0.0, 0.38, 0.0, False),  # its end is still inside the left sleeve
        (0.0, 0.0, 0.40, 0.5, False),  # bar out of both sleeves
        (0.10, 0.0, 0.40, 0.5, False),  # drawer half out
        (0.18, 0.05, 0.40, 1.0, False),  # out 18 cm but still moving
        (0.19, 0.0, 0.40, 1.0, True),  # out and still
    ],
)
def test_scoring(opening, speed, slide, expected_score, expected_success):
    objects = {"cabinet": cabinet_state(opening, speed), "bar": bar_state(slide=slide)}
    assert TASK.score(objects) == expected_score
    assert TASK.success(objects) is expected_success


def test_the_demo_lays_the_bar_on_the_table_clear_of_the_sleeves():
    on_table = bar_state(lift=-(T.BAR_POS[2] - T.TOP - T.BAR_SIZE[2] / 2), slide=T.BAR_OUT)
    assert T.bar_clear(on_table)


def test_the_bar_cannot_leave_the_sleeves_upward():
    assert T.LIP_H - T.BAR_SIZE[2] == pytest.approx(0.005)  # 5 mm of headroom under the sleeve's top
    assert not T.bar_clear(bar_state(lift=T.LIP_H - T.BAR_SIZE[2]))


def test_the_pad_tips_stay_above_the_table_when_the_bar_is_set_down():
    grip_above_bar_centre = T.BAR_GRIP_Z - T.BAR_POS[2]
    tcp_on_table = T.TOP + T.BAR_SIZE[2] / 2 + grip_above_bar_centre + 0.003
    assert tcp_on_table - PAD_TIP_PAST_TCP > T.TOP


def test_the_grip_clears_the_left_sleeve():
    assert T.BAR_GRIP_Y - PAD_HALF_WIDTH > T.BRACKET_Y[1]


def test_the_bar_blocks_only_the_top_drawer_after_its_gap():
    bar_low = np.array(T.BAR_POS) - np.array(T.BAR_SIZE) / 2
    bar_high = np.array(T.BAR_POS) + np.array(T.BAR_SIZE) / 2
    top_front_z = (T.TOP + T.FRONT_Z[2], T.TOP + T.FRONT_Z[2] + T.FRONT_H)
    middle_front_z = (T.TOP + T.FRONT_Z[1], T.TOP + T.FRONT_Z[1] + T.FRONT_H)
    assert bar_low[2] < top_front_z[1] and bar_high[2] > top_front_z[0]  # overlaps the top front in height
    assert bar_low[2] > middle_front_z[1]  # clear above the middle front
    assert bar_low[1] < -T.FRONT_W / 2 and bar_high[1] > T.FRONT_W / 2  # spans the front's full width
    assert T.CAB_X - bar_high[0] == pytest.approx(T.BAR_GAP)  # the drawer can move only the gap before it hits
    assert T.BAR_GAP < T.OPEN_GOAL


def test_the_top_handle_clears_the_bar_and_the_drawer_stays_in_the_cabinet():
    handle_bottom = T.HANDLE_Z - T.HANDLE_H / 2
    assert handle_bottom - (T.TOP + T.BAR_BOTTOM + T.BAR_SIZE[2]) >= 0.01
    assert T.TRAVEL >= T.PULL >= T.OPEN_GOAL


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_demo_is_reachable_within_the_joint_limits(arm):
    plan = ArmPlan(TASK.demo()[arm], station.BASES[arm])
    actions = np.array([plan.action7((k + 1) / CONTROL_HZ)[0] for k in range(round(plan.duration * CONTROL_HZ) + 5)])
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, (plan.worst_ik_mm, plan.worst_ik_deg)
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)


def test_the_demo_fits_the_run_limit():
    longest = max(sum(w.duration for w in waypoints) for waypoints in TASK.demo().values())
    assert longest * CONTROL_HZ < 1200
