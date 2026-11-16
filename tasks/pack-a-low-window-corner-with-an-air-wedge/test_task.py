"""Offline checks (no GPU, no Isaac Sim): the task file loads, the scoring rules, the geometry, and the demo's reach.

    git clone https://github.com/kairosiann/yam-isaac-lab-arena ~/yam-isaac-lab-arena
    PYTHONPATH=~/yam-isaac-lab-arena uv run --python 3.12 --with numpy --with pytest python -m pytest test_task.py
"""

import math
import pathlib
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import task as T  # noqa: E402
from yam_arena import ik, station, task_file  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402

RUN_LIMIT = 2400
FAR = {"left_robot": {"tcp": np.array([0.15, 0.50, T.TOP + 0.16]), "open": 1.0},
       "right_robot": {"tcp": np.array([0.20, -0.20, T.TOP + 0.16]), "open": 0.5}}


def quat(yaw):
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def state(corner_mm=0.0, packer=(T.PX0, T.PY0), packer_yaw=0.0, bag_x=T.WX0, hands=None, packer_z=0.0):
    """corner_mm: the low corner above the sill. packer: its centre (x, y). bag_x: the bag's centre."""
    th = (corner_mm / 1000 - T.PK[2]) / T.LEN
    bag = np.array([bag_x, T.WY0, T.TOP])
    h1 = bag + np.array([-T.BAG_L / 2 - T.BLOCK[0], 0, T.HOSE_Z])
    h2 = h1 + np.array([-T.HOSE / 2, 0, 0])
    bulb = h2 + np.array([-T.HOSE / 2, 0, 0])
    pad = bulb + np.array([-T.NECK - T.BULB_L / 2, 0.015, T.BULB_Z])
    bodies = {"body": bag, "lift": bag, "mid": bag, "h0": h1, "h1": h1, "h2": h2, "bulb": bulb, "pad": pad, "valve": bulb}
    return {"opening": NS(q={"frame": th}),
            "packer": NS(pos=np.array([packer[0], packer[1], T.TOP + packer_z]), quat_xyzw=quat(packer_yaw)),
            "wedge": NS(pos=bag, quat_xyzw=quat(0.0), q={"lift": 0.0, "pad": 0.0, "valve": 0.0}, bodies=bodies,
                        hands=hands or FAR, pumped={})}


UNDER = (T.XF + 0.005, T.PY0)  # the packer pushed in: 5.5 cm of it under the frame
BAG_IN = T.WX0 + T.PUSH
BAG_OUT = T.WX0 + T.PUSH - T.PULL


def test_the_task_file_loads():
    tf = task_file.load(str(HERE / "task.py"))
    assert tf.demo is not None and len(T.OBJECTS) == 3


def test_doing_nothing_scores_zero():
    assert T.score(state()) == 0.0


def test_full_score_needs_the_corner_on_the_packer_the_wedge_out_and_the_hands_away():
    assert T.score(state(6.0, UNDER, bag_x=BAG_OUT)) == 1.0
    assert T.success(state(6.0, UNDER, bag_x=BAG_OUT))


def test_wedge_left_under_the_frame_is_075():
    assert T.score(state(6.0, UNDER, bag_x=BAG_IN)) == 0.75


def test_a_gripper_still_on_the_wedge_or_at_the_frame_is_075():
    on_bulb = dict(FAR, right_robot={"tcp": np.array([BAG_OUT - T.BAG_L / 2 - T.BLOCK[0] - T.HOSE - 0.09, T.WY0, T.TOP + 0.04]), "open": 0.8})
    assert T.score(state(6.0, UNDER, bag_x=BAG_OUT, hands=on_bulb)) == 0.75
    on_block = dict(FAR, right_robot={"tcp": np.array([BAG_OUT - T.BAG_L / 2 - 0.012, T.WY0, T.TOP + 0.017]), "open": 0.2})
    assert T.score(state(6.0, UNDER, bag_x=BAG_OUT, hands=on_block)) == 0.75
    at_frame = dict(FAR, left_robot={"tcp": np.array([T.XF - 0.02, 0.2, T.TOP + 0.2]), "open": 0.0})
    assert T.score(state(6.0, UNDER, bag_x=BAG_OUT, hands=at_frame)) == 0.75
    on_packer = dict(FAR, left_robot={"tcp": np.array([UNDER[0] - 0.06, T.PY0, T.TOP + 0.02]), "open": 0.0})
    assert T.score(state(6.0, UNDER, bag_x=BAG_OUT, hands=on_packer)) == 0.75


def test_packer_in_place_but_the_corner_still_held_up_is_half():
    assert T.score(state(9.5, UNDER, bag_x=BAG_IN)) == 0.5
    assert T.score(state(7.6, UNDER, bag_x=BAG_IN)) == 0.5
    assert T.score(state(7.4, UNDER, bag_x=BAG_IN)) == 0.75


def test_corner_lifted_without_the_packer_under_it_is_a_quarter():
    assert T.score(state(9.5, bag_x=BAG_IN)) == 0.25
    assert T.score(state(3.0, bag_x=BAG_IN)) == 0.25


def test_corner_let_back_down_on_the_sill_scores_zero():
    assert T.score(state(0.0, bag_x=BAG_OUT)) == 0.0
    assert T.score(state(2.0, bag_x=BAG_IN)) == 0.0


@pytest.mark.parametrize("packer, yaw", [
    ((T.XF - 0.02, T.PY0), 0.0),  # only 3 cm of it under the frame
    ((T.XF + 0.005, T.YL - 0.10), 0.0),  # under the bottom rail but 10 cm from the frame's end
    ((T.XF + 0.005, T.YL + 0.02), 0.0),  # beside the frame, in the gap at its end
    ((T.XF + 0.005, T.PY0), math.radians(40)),  # skewed
])
def test_a_packer_beside_or_short_of_the_corner_does_not_count(packer, yaw):
    assert not T.packer_under(state(9.5, packer, yaw))
    assert T.score(state(9.5, packer, yaw, bag_x=BAG_IN)) == 0.25


def test_a_packer_lying_on_something_does_not_count():
    assert not T.packer_under(state(9.5, UNDER, packer_z=0.01))


def test_the_window_starts_six_mm_low_and_can_reach_the_head():
    assert T.corner(state()) == pytest.approx(0.0)
    assert T.PK[2] + T.LEN * T.FRAME.high == pytest.approx(T.PK[2] + T.HEAD_GAP)
    assert T.FRAME.low == pytest.approx(-T.PK[2] / T.LEN)


def test_the_flat_bag_fits_under_the_rail_where_the_demo_slides_it_in():
    for y in (T.WY0 - T.BAG_W / 2, T.WY0 + T.BAG_W / 2):
        gap = T.PK[2] * (1 - (y - T.PIV_Y) / T.LEN)
        assert gap > T.BAG_T + 0.001, (y, gap)


def test_the_loose_packer_does_not_fit_until_the_corner_is_lifted():
    gap_at_packer = T.PK[2] * (1 - (T.PY0 - T.PK[1] / 2 - T.PIV_Y) / T.LEN)
    assert gap_at_packer < T.PK[2] / 4


def test_the_demos_squeezes_lift_the_corner_clear_of_the_packer():
    d = T.WY0 - T.PIV_Y  # the bag's middle from the pivot
    slack = T.PK[2] * (1 - d / T.LEN) - T.BAG_T
    sag = T.MASS * 9.81 * (0.5 * (T.YR + T.YL) - T.PIV_Y) / d / T.LIFT.stiffness
    corner = T.PK[2] * 0 + (T.SQUEEZES * T.STEP - slack - sag) * T.LEN / d
    assert T.PK[2] + 0.002 < corner < T.PK[2] + T.HEAD_GAP + 0.001, corner


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_demo_is_reachable_within_the_joint_limits(arm):
    plan = ArmPlan(T.demo()[arm], station.BASES[arm])
    actions = np.array([plan.action7((k + 1) / 20)[0] for k in range(round(plan.duration * 20))])
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, (plan.worst_ik_mm, plan.worst_ik_deg)
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)


def test_the_demo_fits_the_run_limit():
    assert max(sum(w.duration for w in plan) for plan in T.demo().values()) * 20 < RUN_LIMIT / 2


def test_the_demo_ends_with_both_grippers_clear():
    d = T.demo()
    hands = {f"{a}_robot": {"tcp": np.array(d[a][-1].pos), "open": 0.0} for a in ("left", "right")}
    assert T.hands_clear(state(6.0, UNDER, bag_x=BAG_OUT, hands=hands))
