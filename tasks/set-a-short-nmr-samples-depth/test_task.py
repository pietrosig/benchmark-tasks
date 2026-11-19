"""Offline checks of the scorer (no simulator): the start scores 0, the right end state scores 1.0, and the bypasses in
our review checklist score 0 or are capped. Run: python -m pytest -q test_task.py"""

from __future__ import annotations

import math
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import task as T  # noqa: E402

UP = np.array([0.0, 0.0, 0.0, 1.0])
FAR = {"left_robot": {"tcp": np.array([0.15, 0.30, T.TOP + 0.25]), "open": 1.0},
       "right_robot": {"tcp": np.array([0.15, -0.30, T.TOP + 0.25]), "open": 1.0}}


def _tilted(deg, axis=(1.0, 0.0, 0.0)):
    h = math.radians(deg) / 2
    a = np.array(axis) * math.sin(h)
    return np.array([a[0], a[1], a[2], math.cos(h)])


def obj(pos, quat=UP, hands=None, contacts=None, q=None):
    return SimpleNamespace(pos=np.array(pos, float), quat_xyzw=np.array(quat, float), lin_vel=np.zeros(3),
                           ang_vel=np.zeros(3), hands=hands or FAR, contacts=contacts if contacts is not None else {"body": []},
                           q=q or {})


def scene(spinner_xy=T.SLOT_XY, spinner_z=T.SEAT + 0.0003, err=0.0, hands=None, contacts=None, tube_pos=None,
          tube_quat=UP, nb_shift=0.0):
    """The spinner seated at ``spinner_xy``, the tube through it with its column ``err`` m below the line."""
    b = T.D_LINE + T.COL_MID + err  # tube bottom below the shoulder
    tp = tube_pos if tube_pos is not None else (*spinner_xy, T.TOP + spinner_z - b)
    h = hands or FAR
    return {
        "spinner": obj((*spinner_xy, T.TOP + spinner_z), hands=h, contacts={"body": []}),
        "tube": obj(tp, tube_quat, hands=h, contacts=contacts if contacts is not None else {"body": []}),
        "nb_right": obj((T.NB_XY["nb_right"][0] + nb_shift, T.NB_XY["nb_right"][1], T.TOP + T.SEAT + 0.0003)),
        "nb_left": obj((*T.NB_XY["nb_left"], T.TOP + T.SEAT + 0.0003)),
    }


@pytest.fixture(autouse=True)
def fresh():
    T.reset_credit()
    yield


def run(states):
    s = 0.0
    for st in states:
        s = T.score(st)
    return s


def test_start_scores_zero():
    st = scene(spinner_xy=T.GAUGE_XY, tube_pos=(*T.RACK_XY, T.TOP + 0.0005))
    assert run([st]) == 0.0


def test_full_score_in_the_carousel():
    assert run([scene(err=0.0005)]) == 1.0
    assert T.success(scene(err=0.0005))


def test_set_in_the_gauge_only_is_075():
    assert run([scene(spinner_xy=T.GAUGE_XY, err=-0.0008)]) == 0.75


def test_depth_bands():
    assert run([scene(err=0.002)]) == 0.5
    T.reset_credit()
    assert run([scene(err=-0.0029)]) == 0.5
    T.reset_credit()
    assert run([scene(err=0.006)]) == 0.25


def test_pushed_to_the_gauge_floor_is_at_most_025():
    # tube bottom on the floor stop: the column is 4 mm low
    floor_err = (T.SEAT - T.FLOOR) - (T.D_LINE + T.COL_MID)
    assert abs(floor_err - 0.004) < 1e-9
    assert run([scene(spinner_xy=T.GAUGE_XY, err=floor_err)]) == 0.25


def test_spinner_beside_the_slot_scores_zero():
    # put down 5 mm off the position's axis (resting on the plate, not in the seat), or 4 mm high
    assert run([scene(spinner_xy=(T.SLOT_XY[0] + 0.005, T.SLOT_XY[1]))]) == 0.0
    T.reset_credit()
    assert run([scene(spinner_z=T.SEAT + 0.004)]) == 0.0


def test_hand_still_holding_scores_zero():
    h = dict(FAR)
    h["left_robot"] = {"tcp": np.array([*T.SLOT_XY, T.TOP + T.SEAT + 0.009]), "open": 0.24}
    assert run([scene(hands=h)]) == 0.0


def test_lower_tube_touch_latches_zero():
    # a finger on the tube 60 mm above its bottom, at any time, then a perfect end
    b = T.D_LINE + T.COL_MID
    tcp = np.array([*T.GAUGE_XY, T.TOP + T.SEAT - b + 0.060])
    h = dict(FAR)
    h["right_robot"] = {"tcp": tcp, "open": 0.05}
    bad = scene(spinner_xy=T.GAUGE_XY, hands=h, contacts={"body": ["right_robot/tip_left"]})
    assert run([bad, scene(err=0.0)]) == 0.0


def test_sleeve_grip_is_fine():
    b = T.D_LINE + T.COL_MID
    tcp = np.array([*T.GAUGE_XY, T.TOP + T.SEAT - b + 0.155])
    h = dict(FAR)
    h["right_robot"] = {"tcp": tcp, "open": 0.05}
    ok = scene(spinner_xy=T.GAUGE_XY, hands=h, contacts={"body": ["right_robot/tip_left", "right_robot/tip_right"]})
    assert run([ok, scene(err=0.0)]) == 1.0


def test_other_robot_link_on_tube_latches_zero():
    st = scene(spinner_xy=T.GAUGE_XY, contacts={"body": ["left_robot/link5"]})
    assert run([st, scene()]) == 0.0


def test_carried_by_the_tube_latches_zero():
    # spinner 50 mm above the gauge with the tube through it and no gripper on its body
    b = T.D_LINE + T.COL_MID
    lifted = scene(spinner_xy=T.GAUGE_XY, spinner_z=T.SEAT + 0.05)
    lifted["tube"].pos = np.array([*T.GAUGE_XY, T.TOP + T.SEAT + 0.05 - b])
    assert run([lifted] * T.UNHELD_STEPS + [scene()]) == 0.0


def test_hand_near_but_not_touching_spinner_is_not_holding():
    # tool point at the turbine with the fingers at its width, but no contact: still carried by the tube
    b = T.D_LINE + T.COL_MID
    h = dict(FAR)
    h["left_robot"] = {"tcp": np.array([*T.GAUGE_XY, T.TOP + T.SEAT + 0.05 + T.TURB_H / 2]), "open": 0.24}
    lifted = scene(spinner_xy=T.GAUGE_XY, spinner_z=T.SEAT + 0.05, hands=h)
    lifted["tube"].pos = np.array([*T.GAUGE_XY, T.TOP + T.SEAT + 0.05 - b])
    assert run([lifted] * T.UNHELD_STEPS + [scene()]) == 0.0


def test_carried_by_the_spinner_is_fine():
    b = T.D_LINE + T.COL_MID
    h = dict(FAR)
    h["left_robot"] = {"tcp": np.array([*T.GAUGE_XY, T.TOP + T.SEAT + 0.05 + T.TURB_H / 2]), "open": 0.24}
    lifted = scene(spinner_xy=T.GAUGE_XY, spinner_z=T.SEAT + 0.05, hands=h)
    lifted["spinner"].contacts = {"body": ["left_robot/tip_left", "left_robot/tip_right"]}
    lifted["tube"].pos = np.array([*T.GAUGE_XY, T.TOP + T.SEAT + 0.05 - b])
    assert run([lifted] * 20 + [scene()]) == 1.0


def test_knocked_neighbour_latches_zero():
    assert run([scene(nb_shift=0.004), scene()]) == 0.0


def test_dropped_tube_put_back_scores_zero():
    dropped = scene()
    dropped["tube"].quat_xyzw = _tilted(80)
    assert run([dropped, scene()]) == 0.0


def test_tube_not_through_scores_zero():
    # the tube standing beside the seated spinner, at the right height
    st = scene()
    st["tube"].pos = st["tube"].pos + np.array([0.0, 0.004, 0.0])
    assert run([st]) == 0.0


def test_reaching_then_undoing_does_not_keep_credit():
    # a perfect set, then pulled 6 mm high: the end state counts
    assert run([scene(err=0.0), scene(err=-0.006)]) == 0.25


def test_demo_reaches_its_targets():
    plan = T.demo()
    assert set(plan) == {"left", "right"}
    total = max(sum(w.duration for w in v) for v in plan.values())
    assert total * 20 < 2400  # inside the run limit (steps at 20 Hz)
