"""Offline checks of the scorer (no simulator): the start scores 0, the right end state scores 1.0, every outcome in the
scoring text gets its stated score, and the review checklist's seven bypasses score 0 or are capped.
Run: python -m pytest -q test_task.py"""

from __future__ import annotations

import math
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import task as T  # noqa: E402

FLAT = np.array([0.0, 0.0, 0.0, 1.0])
FAR = {"left_robot": {"tcp": np.array([0.18, 0.30, T.TOP + 0.25]), "open": 1.0},
       "right_robot": {"tcp": np.array([0.15, -0.30, T.TOP + 0.25]), "open": 1.0}}


def q_axis(axis, deg):
    h = math.radians(deg) / 2
    a = np.array(axis, float) * math.sin(h)
    return np.array([a[0], a[1], a[2], math.cos(h)])


DOWN = q_axis((0, 1, 0), -90)  # the head end (the knife's point, at world -x when flat) pointing down
UP = q_axis((0, 1, 0), 90)  # the head end pointing up (the knife's point up)


def obj(pos, quat=FLAT, hands=None, held=False, vel=0.0):
    links = ["left_robot/tip_left"] if held else []
    return SimpleNamespace(pos=np.array(pos, float), quat_xyzw=np.array(quat, float), lin_vel=np.array([vel, 0, 0]),
                           ang_vel=np.zeros(3), hands=hands or FAR, contacts={"body": links}, q={})


def on_board(name="knife", hands=None):
    return obj((T.ITEM_X[name], T.ITEM_Y[name], T.TOP + T.ITEM_Z[name]), hands=hands)


def in_cell(name, cell, head_down=True, z_low=None, hands=None, held=False, tilt=0.0):
    """The piece upright in ``cell``, its lower end ``z_low`` above the bench (default: on the basket floor)."""
    lo, hi = T.ENDS[name]
    q = DOWN if head_down else UP
    if tilt:
        q = q_mul(q_axis((1, 0, 0), tilt), q)
    rot = T._rot(q)
    a = -rot[:, 0]
    z_low = T.B_FLOOR + 0.001 if z_low is None else z_low
    # put the lower end at (cell, z_low)
    low_local = hi if head_down else lo
    p = np.array([cell[0], cell[1], z_low]) - low_local * a
    return obj((p[0], p[1], p[2] + T.TOP), q, hands=hands, held=held)


def q_mul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def scene(knife=None, hands=None):
    h = hands or FAR
    st = {"knife": knife or on_board(hands=h)}
    st["knife"].hands = h
    return st


OPEN_M = (T.CELLS[1][0] + T.OPEN_DX, T.CELLS[1][1])  # the front middle cell's opening (the demo's)
OPEN_C = (T.CELLS[4][0] + T.OPEN_DX, T.CELLS[4][1])  # another free cell's opening


def hung(**kw):
    return scene(knife=in_cell("knife", OPEN_M), **kw)


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
    assert run([scene()]) == 0.0


def test_full_score_knife_point_down_in_its_cell():
    assert run([hung()]) == 1.0
    assert T.success(hung())


def test_knife_point_down_in_another_cell_counts():
    assert run([scene(knife=in_cell("knife", OPEN_C))]) == 1.0


def test_seated_knife_is_full_and_its_whole_edge_is_below_the_lid():
    k = in_cell("knife", OPEN_M)
    butt, point = T.ends("knife", k)
    assert T.seated(k)
    assert T.heel(k)[2] < T.LID_UNDER
    assert butt[2] - T.LID_Z <= T.HANDLE_ABOVE_MAX


def _upright_with_butt_at(above_lid, tilt=0.0):
    """Upright point down in the demo's cell, lowered so the handle's end is ``above_lid`` over the lid's top."""
    k = in_cell("knife", OPEN_M, tilt=tilt)
    butt, _ = T.ends("knife", k)
    return in_cell("knife", OPEN_M, tilt=tilt, z_low=T.B_FLOOR + 0.001 + (T.LID_Z + above_lid - butt[2]))


def test_handle_95_mm_boundary():
    assert run([scene(knife=_upright_with_butt_at(0.0945))]) == 1.0
    T.reset_credit()
    assert run([scene(knife=_upright_with_butt_at(0.0955))]) == 0.25


def test_tilt_20_with_the_point_20_mm_up_is_partial():
    k = in_cell("knife", OPEN_M, tilt=20.0, z_low=T.B_FLOOR + 0.020)
    assert run([scene(knife=k)] * 6) == 0.25


def test_tilt_21_degrees_seated_is_partial():
    k = in_cell("knife", OPEN_M, tilt=21.0)
    assert not T.seated(k)
    assert run([scene(knife=k)] * 6) == 0.25


def test_rule_implies_edge_below_lid_over_a_sweep():
    # every released pose the rule accepts has its heel, plus the 8 mm edge corner at tilt, below the lid's underside
    import math
    for tilt in range(0, 21, 2):
        for dz in range(0, 40, 2):
            k = in_cell("knife", OPEN_M, tilt=float(tilt), z_low=T.B_FLOOR + 0.001 + dz / 1000)
            if T.seated(k):
                corner = T.heel(k)[2] + 0.008 * math.sin(math.radians(tilt))
                assert corner < T.LID_UNDER


@pytest.mark.parametrize("below_lid", [0.010, 0.020, 0.030])
def test_caught_partway_point_down_with_edge_exposed_is_not_full(below_lid):
    k = in_cell("knife", OPEN_M, z_low=T.LID_Z - below_lid)
    assert not T.seated(k) and T.heel(k)[2] > T.LID_UNDER
    assert run([scene(knife=k)] * 6) == 0.25


def test_hung_in_the_opening_halfway_is_not_full():
    k = in_cell("knife", OPEN_M, z_low=T.B_FLOOR + 0.040)
    assert not T.seated(k)
    assert run([scene(knife=k)] * 6) == 0.25


def test_knife_point_down_leaning_6_degrees_counts():
    assert run([scene(knife=in_cell("knife", OPEN_C, tilt=6.0))]) == 1.0


def test_knife_point_up_in_a_cell_is_zero():
    assert run([scene(knife=in_cell("knife", OPEN_C, head_down=False))] * 6) == 0.0


def test_point_up_threshold_is_20_degrees_from_level():
    k30 = obj((0.40, -0.10, T.TOP + T.RACK_FLOOR + 0.06), q_axis((0, 1, 0), 30))  # point up 30 degrees
    k10 = obj((0.40, -0.10, T.TOP + T.RACK_FLOOR + 0.03), q_axis((0, 1, 0), 10))  # point up 10 degrees
    assert T.point_up(k30) and not T.point_up(k10)
    assert abs(T.elevation_deg(k30) - 30.0) < 0.5
    assert run([scene(knife=k30)] * 6) == 0.0


def test_knife_resting_point_up_25_degrees_on_the_lid_is_zero():
    k = obj((T.CELLS[1][0], T.CELLS[1][1], T.TOP + T.LID_Z + 0.05), q_axis((0, 1, 0), 25))
    assert run([scene(knife=k)] * 6) == 0.0


def test_knife_left_on_the_board_is_zero():
    assert run([scene()] * 6) == 0.0


def test_knife_flat_on_the_lid_is_partial():
    k = obj((T.CELLS[1][0] + 0.03, T.CELLS[1][1], T.TOP + T.LID_Z + 0.007))
    assert run([scene(knife=k)] * 6) == 0.25


def test_knife_on_the_lid_point_slightly_up_is_still_partial():
    k = obj((T.CELLS[1][0] + 0.03, T.CELLS[1][1], T.TOP + T.LID_Z + 0.02), q_axis((0, 1, 0), 8))
    assert run([scene(knife=k)] * 6) == 0.25


def test_knife_loose_on_the_rack_floor_is_zero():
    k = obj((0.40, -0.10, T.TOP + T.RACK_FLOOR + 0.007))
    assert run([scene(knife=k)] * 6) == 0.0


def test_knife_wedged_across_a_cell_point_down_is_zero():
    k = in_cell("knife", OPEN_C, tilt=60.0, z_low=T.B_FLOOR + 0.030)
    assert not T.point_down_loaded(k)
    assert run([scene(knife=k)] * 6) == 0.0


def test_knife_on_the_bench_off_the_door_is_zero():
    assert run([scene(knife=obj((0.10, -0.40, T.TOP + 0.007)))] * 6) == 0.0


def test_let_go_point_up_over_the_rack_latches_even_if_it_then_lands_point_down():
    hold = in_cell("knife", OPEN_C, head_down=False, z_low=T.LID_Z + 0.02, held=True)
    drop = in_cell("knife", OPEN_C, head_down=False, z_low=T.LID_Z + 0.01, held=False)
    assert run([scene(knife=hold), scene(knife=drop), hung()]) == 0.0


def test_held_point_up_in_transit_then_loaded_point_down_is_full():
    hold = in_cell("knife", OPEN_C, head_down=False, z_low=T.LID_Z + 0.05, held=True)
    assert run([scene(knife=hold), scene(knife=hold), hung()]) == 1.0


def test_hands_not_clear_is_zero():
    near = {"left_robot": {"tcp": np.array([OPEN_M[0], OPEN_M[1], T.TOP + T.LID_Z + 0.02]), "open": 1.0},
            "right_robot": FAR["right_robot"]}
    assert run([hung(hands=near)]) == 0.0


# -- the seven bypass tests ---------------------------------------------------------------------------------------


def test_bypass_1_do_nothing():
    assert run([scene()] * 30) == 0.0


def test_bypass_2_pushed_into_the_rack_instead_of_loaded():
    k = obj((0.41, 0.05, T.TOP + T.RACK_FLOOR + 0.007))  # shoved off the board into the rack, lying flat
    assert run([scene(knife=k)] * 6) == 0.0


def test_bypass_3_loaded_point_down_then_turned_point_up():
    up = scene(knife=in_cell("knife", OPEN_C, head_down=False))
    assert run([hung()] * 3 + [up] * 6 + [hung()]) == 0.0


def test_bypass_4_beside_the_basket_point_down_is_not_in_a_cell():
    out = (T.BASKET_X[1] + 0.03, T.CELLS[5][1])  # upright, point down, on the rack floor behind the basket
    assert run([scene(knife=in_cell("knife", out, z_low=T.RACK_FLOOR + 0.001))] * 6) == 0.0


def test_bypass_5_dropped_off_the_bench_then_put_back_is_zero():
    assert run([scene(knife=obj((0.30, 0.55, T.TOP - 0.30))), hung()]) == 0.0


def test_bypass_5b_dropped_on_the_bench_and_picked_up_again_is_fine():
    assert run([scene(knife=obj((0.10, 0.20, T.TOP + 0.007))), hung()]) == 1.0


def test_seated_then_pulled_up_partway_is_not_full():
    assert run([hung()] * 3 + [scene(knife=in_cell("knife", OPEN_M, z_low=T.LID_Z - 0.02))] * 3) == 0.25


def test_bypass_6_proximity_is_not_loading():
    # point down just over the opening, not through it (held there, so also not clear): 0
    assert run([scene(knife=in_cell("knife", OPEN_M, z_low=T.LID_Z + 0.005))]) == 0.0


def test_bypass_7_another_robot_link_touching_does_not_matter():
    st = hung()
    st["knife"].contacts = {"body": ["right_robot/link_5"]}
    assert run([st]) == 1.0
