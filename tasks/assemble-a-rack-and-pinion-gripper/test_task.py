"""Offline checks (no simulator): the mesh geometry, and the scorer on made-up states.

    PYTHONPATH=<yam-isaac-lab-arena> python -m pytest -q test_task.py
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
    sys.modules.pop("task", None)
    return importlib.import_module("task")


FAR_HANDS = {"left_robot": {"tcp": np.array([0.0, 0.3, 1.0]), "open": 1.0},
             "right_robot": {"tcp": np.array([0.0, -0.3, 1.0]), "open": 1.0}}
UP = (0.0, 0.0, 0.0, 1.0)


def _yaw_q(a):
    return (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))


def scene(task, step, *, sn, sf, far_seated=True, near_seated=True, hands=None, touch=None, far_pos=None, tilt=0.0):
    """Racks at openings sn, sf (m) in an unmoved housing."""
    hands = hands or FAR_HANDS
    touch = touch or {}
    h = SimpleNamespace(pos=np.array([task.XH, task.YH, task.TOP]), quat_xyzw=np.array(UP), hands=hands,
                        contacts={"body": touch.get("housing", []), "pinion": []}, step=step, q={"pinion": 0.0})
    def rack(side, s, seated_, name):
        p = np.array([task.XH + side * task.R, task.YH + side * (task.CLOSED_C + s), task.TOP + task.RB])
        if not seated_:
            p = p + np.array([0.0, 0.0, 0.004])  # resting on the pin tips
        q = (math.sin(tilt / 2), 0.0, 0.0, math.cos(tilt / 2)) if tilt else UP
        return SimpleNamespace(pos=p if far_pos is None or name != "far" else np.array(far_pos), quat_xyzw=np.array(q),
                               hands=hands, contacts={"body": touch.get(name, [])}, step=step, q={})
    return {"housing": h, "near": rack(-1, sn, near_seated, "near"), "far": rack(1, sf, far_seated, "far")}


def run(task, states):
    s = None
    for st in states:
        s = task.score(st)
    return s


def test_mesh_has_clearance_at_every_angle(task):
    """Pins never cut into a tooth while the pinion turns, with the racks at the right tooth or one off."""
    R = task.R
    def teeth(side, s):
        out = []
        for (c, z, *_rest) in task._rack_boxes(side, (0, 0, 0)):
            if abs(z[1] - task.TW) < 1e-9 and c[2] > 0.002:
                x, y = c[0] + side * R, c[1] + side * (task.CLOSED_C + s)
                out.append((x - z[0] / 2, x + z[0] / 2, y - z[1] / 2, y + z[1] / 2))
        return out
    for idx in (0, 1, -1):
        worst = 1.0
        for th in np.linspace(0, math.pi * 3 / 4, 200):
            s = task.START_OPEN + R * th
            for k in range(8):
                a = th + k * math.pi / 4
                px, py = R * math.cos(a), R * math.sin(a)
                for side, ss in ((-1, s), (1, s + idx * task.P)):
                    for x0, x1, y0, y1 in teeth(side, ss):
                        worst = min(worst, math.hypot(max(x0 - px, 0, px - x1), max(y0 - py, 0, py - y1)) - task.PIN_D / 2)
        assert worst > 0.0001


def test_backlash_is_under_half_a_millimetre(task):
    assert task.GAP - task.PIN_D <= 0.0005


def test_one_tooth_is_4_7_mm_of_asymmetry(task):
    assert task.P / 2 == pytest.approx(0.00471, abs=1e-5)


def test_nothing_done_scores_zero(task):
    st = scene(task, 0, sn=task.START_OPEN, sf=0.0, far_pos=task.FAR0)
    assert run(task, [st]) == 0.0


@pytest.mark.parametrize("bl", [-0.0004, 0.0, 0.0004])  # the two backlash extremes (0.2 mm each rack, opposite ways)
def test_right_tooth_passes_at_backlash_extremes(task, bl):
    s0 = task.START_OPEN
    assert run(task, [scene(task, 0, sn=s0, sf=s0 + bl)]) == 0.5


@pytest.mark.parametrize("teeth", [1, -1, 2])
def test_wrong_tooth_scores_zero(task, teeth):
    s0 = 2 * task.P
    assert run(task, [scene(task, 0, sn=s0, sf=s0 + teeth * task.P)]) == 0.0


def _opening_run(task, *, touch_far=False, touch_both=False, steps=60):
    states = []
    for k in range(steps):
        s = task.START_OPEN + 0.0175 * min(1.0, k / 40)
        touch = {}
        if k < 40:
            touch["near"] = ["right_robot/tip_left"]
            if touch_far or touch_both:
                touch["far"] = ["left_robot/tip_left"]
        states.append(scene(task, k, sn=s, sf=s, touch=touch))
    return run(task, states)


def test_opened_by_the_white_rack_alone_scores_one(task):
    assert _opening_run(task) == 1.0


def test_both_racks_dragged_by_hand_scores_half(task):
    assert _opening_run(task, touch_both=True) == 0.5


def test_open_but_not_far_enough(task):
    states = [scene(task, k, sn=task.START_OPEN + 0.004 * k / 30, sf=task.START_OPEN + 0.004 * k / 30) for k in range(40)]
    assert run(task, states) == 0.5


def test_unseated_rack_scores_zero(task):
    s0 = task.START_OPEN
    assert run(task, [scene(task, 0, sn=s0, sf=s0, far_seated=False)]) == 0.0
    assert run(task, [scene(task, 1, sn=s0, sf=s0, near_seated=False)]) == 0.0


def test_near_rack_lifted_out_beside_the_housing_scores_zero(task):
    s0 = 0.02
    st = scene(task, 0, sn=s0, sf=s0)
    st["near"].pos = st["near"].pos + np.array([-0.049, 0.0, -task.RB])
    assert run(task, [st]) == 0.0


def test_tilted_rack_scores_zero(task):
    s0 = task.START_OPEN
    assert run(task, [scene(task, 0, sn=s0, sf=s0, tilt=math.radians(5))]) == 0.0


def test_gripper_still_near_the_jaws_scores_half(task):
    near_hand = dict(FAR_HANDS, right_robot={"tcp": np.array([task.XH - 0.018, -0.03, task.TOP + 0.05]), "open": 1.0})
    states = []
    for k in range(60):
        s = task.START_OPEN + 0.0175 * min(1.0, k / 40)
        states.append(scene(task, k, sn=s, sf=s, hands=near_hand if k >= 40 else None,
                            touch={"near": ["right_robot/tip_left"]} if k < 40 else {}))
    assert run(task, states) == 0.5


def test_dragging_the_orange_rack_does_not_count(task):
    states = []
    for k in range(60):
        s = task.START_OPEN + 0.0175 * min(1.0, k / 40)
        states.append(scene(task, k, sn=s, sf=s, touch={"far": ["left_robot/tip_left"]} if k < 40 else {}))
    assert run(task, states) == 0.5


def test_record_is_cleared_on_a_new_episode(task):
    assert _opening_run(task) == 1.0
    assert run(task, [scene(task, 0, sn=task.START_OPEN, sf=task.START_OPEN)]) == 0.5
    assert task._H["coupled"] == 0.0
