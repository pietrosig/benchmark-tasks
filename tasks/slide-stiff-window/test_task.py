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
RESET = TASK.score.__globals__["reset_credit"]  # the loaded task's own counter (``task`` imported above is a second copy)


HOLD = ["right_robot/tip_left", "right_robot/tip_right"]  # the right gripper's fingers on the handle's grip bar


def window(opened=0.0, give=0.0, torn=False, speed=0.0, hand="on", fingers=0.13, push=0.0, other=None, touch=None,
           contacts="auto"):
    """hand: "on" (a gripper's tool point at the grip bar), "off" (far away) or "push" (at the sash's far edge).
    push: the force (N) something other than the handle puts on the sash in the opening direction (the track's friction,
    the joint's drive, balances the rest, with the sash not accelerating).
    touch: extra robot contacts PhysX reports, {"body" | "sash" | "handle": ["arm/link", ...]}. With contacts="auto" the
    right fingers touch the handle when the hand is "on" (closed on it), and its fingertip touches the sash when "push".
    contacts=None: PhysX reported nothing (the contact view failed)."""
    p = np.array(T.WINDOW.pos, dtype=float)
    q = {"sash": -opened, "handle": -give}
    bar = np.array([T.X + T.SASH_FRONT + T.SASH_D / 2 + T._FACE - T.STAND - T.BAR_X / 2,
                    T.SASH_Y + T.HANDLE_Y - opened - give, T.TOP + T.SILL + T.GAP_Z + T.HANDLE_Z])
    tcp = {"on": bar, "off": np.array([0.15, -0.3, T.TOP + 0.2]), "push": bar + np.array([0.0, T.SASH_W, 0.0])}[hand]
    left = np.array([0.15, 0.3, T.TOP + 0.2]) if other is None else np.asarray(other, dtype=float)
    hands = {"right_robot": {"tcp": tcp, "open": fingers}, "left_robot": {"tcp": left, "open": 1.0}}
    if contacts == "auto":
        contacts = {"body": [], "sash": [], "handle": []}
        if hand == "on" and fingers < 0.3:
            contacts["handle"] += HOLD
        if hand == "push":
            contacts["sash"] += ["right_robot/tip_left"]
        for part, links in (touch or {}).items():
            contacts[part] = contacts[part] + list(links)
    return {"window": FixtureState(pos=p, quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                                   low=p, high=p, q=q, speed={"sash": -speed, "handle": 0.0},
                                   broken={"sash": False, "handle": torn}, hands=hands,
                                   drive={"sash": T.HANDLE_K * give + push, "handle": 0.0}, accel={"sash": 0.0, "handle": 0.0},
                                   contacts=contacts)}


def trial(*states):
    """Score a sequence of states from a fresh start (the score counts travel step by step); returns the last score."""
    RESET()
    out = None
    for st in (window(), *states):
        out = TASK.score(st)
    return out


def pulled(opened, give=0.006, **kw):
    """A state pulled open to ``opened`` by the gripped handle (6 mm of give is about 5 N)."""
    return window(opened, give=give, **kw)


def test_the_task_file_loads():
    assert TASK.scene == "station"
    assert [o.name for o in TASK.objects] == ["window"]
    assert all(callable(f) for f in (TASK.success, TASK.distance, TASK.score, TASK.demo))


@pytest.mark.parametrize("opened, expected", [(0.0, 0.0), (0.039, 0.0), (0.04, 0.25), (0.079, 0.25), (0.08, 0.5),
                                              (0.119, 0.5), (0.12, 0.75), (0.149, 0.75), (0.15, 1.0), (0.24, 1.0)])
def test_the_score_follows_how_far_the_sash_is_pulled_open(opened, expected):
    assert trial(pulled(opened)) == expected


def test_doing_nothing_scores_zero():
    assert trial(window()) == 0.0


@pytest.mark.parametrize("opened", [0.0, 0.06, 0.15, 0.24])
def test_a_torn_handle_scores_zero(opened):
    assert trial(pulled(opened, give=0.14, torn=True)) == 0.0


def test_a_handle_under_strain_but_on_still_counts():
    assert trial(pulled(0.18, give=0.015)) == 1.0


def test_pushing_the_sash_open_earns_nothing():
    assert trial(window(0.20, hand="push", fingers=1.0)) == 0.0
    assert trial(window(0.20, hand="push", fingers=0.0)) == 0.0  # closed fingers pushing the sash's edge


def test_pushing_the_handle_without_gripping_it_earns_nothing():
    assert trial(window(0.20, give=0.006, fingers=1.0)) == 0.0  # open fingers
    assert trial(window(0.20, give=0.006, fingers=0.0)) == 0.0  # shut, empty fingers pushing the bar


def test_a_gripped_handle_that_carries_no_pull_earns_nothing():
    assert trial(window(0.20, give=0.0)) == 0.0  # e.g. the other hand pushes the sash while one holds the handle


def test_only_the_pulled_travel_counts():
    # pushed 10 cm, then pulled 4 cm by the handle: 4 cm count
    assert trial(window(0.10, hand="push"), pulled(0.14)) == 0.25
    # pulled 10 cm, then let go and it is shoved 10 cm more: 10 cm count
    assert trial(pulled(0.10), window(0.20, hand="off", fingers=1.0)) == 0.5


def test_one_gripper_lightly_tensioning_the_handle_while_the_other_pushes_earns_nothing():
    # the handle carries 1.2 N (1.5 mm of give, enough to count as pulling on its own) while another gripper pushes 6 N
    assert trial(window(0.20, give=0.0015, push=6.0)) == 0.0
    # in small steps, the same
    assert trial(*[window(0.01 * k, give=0.0015, push=6.0) for k in range(1, 21)]) == 0.0
    # and a weaker push that still does most of the work
    assert trial(*[window(0.01 * k, give=0.002, push=3.0) for k in range(1, 21)]) == 0.0


AWAY = (0.20, 0.30, T.TOP + 0.10)  # the left tool point well outside the window, 10 cm and more from it


@pytest.mark.parametrize("touch", [
    {"body": ["left_robot/link4"]},  # the left forearm resting on the frame
    {"sash": ["left_robot/link5"]},  # the left wrist leaning on the sash
    {"sash": ["left_robot/gripper"]},  # the back of the left gripper on the sash's glass, its fingertips away
    {"body": ["right_robot/link5"]},  # the gripping arm's own wrist pressing the frame while it pulls
    {"sash": ["right_robot/gripper"]},  # the gripping hand's knuckles on the sash while its fingers hold the bar
    {"handle": ["left_robot/tip_left"]},  # the other gripper pushing the handle along
])
def test_another_robot_part_touching_the_window_earns_nothing_wherever_its_tool_point_is(touch):
    # no force on the sash besides the handle's pull, and the left tool point far from the window: only the contact
    # PhysX reports between a robot link and the window shows the touch
    assert trial(*[pulled(0.01 * k, give=0.009, other=AWAY, touch=touch) for k in range(1, 21)]) == 0.0


def test_a_touch_only_stops_credit_while_it_lasts():
    seq = [pulled(0.01 * k, give=0.009, other=AWAY) for k in range(1, 9)]  # 8 cm pulled cleanly
    seq += [pulled(0.01 * k, give=0.009, other=AWAY, touch={"body": ["left_robot/link4"]}) for k in range(9, 15)]
    seq += [pulled(0.01 * k, give=0.009, other=AWAY) for k in range(15, 21)]  # 6 cm more, cleanly
    assert trial(*seq) == 0.75  # 8 + 6 = 14 cm counted, not 20


def test_unknown_contacts_count_nothing():
    assert trial(*[pulled(0.01 * k, give=0.009, contacts=None) for k in range(1, 21)]) == 0.0


def test_the_other_gripper_away_from_the_window_does_not_matter():
    assert trial(*[pulled(0.01 * k, give=0.009, other=AWAY) for k in range(1, 21)]) == 1.0


def test_the_other_gripper_near_the_window_without_touching_it_does_not_matter():
    near = (T.X + T.SASH_FRONT - 0.01, T.SASH_Y, T.TOP + T.SILL + 0.10)  # 1 cm in front of the sash, not touching
    assert trial(*[pulled(0.01 * k, give=0.009, other=near) for k in range(1, 21)]) == 1.0


def test_a_little_force_noise_does_not_cancel_a_real_pull():
    assert trial(*[pulled(0.01 * k, give=0.009, push=1.0) for k in range(1, 21)]) == 1.0  # 7 N on the handle, 1 N elsewhere


def test_short_pulls_closed_in_between_then_a_push_cannot_earn_credit():
    seq = []
    for _ in range(4):  # pull 5 cm by the handle, then close it (any way), four times: 20 cm pulled in all
        seq += [pulled(0.05), window(0.0, hand="off", fingers=1.0)]
    seq += [window(0.01 * k, hand="push", fingers=1.0, push=7.0) for k in range(1, 21)]  # then push it 20 cm open
    assert trial(*seq) == 0.0


def test_short_pulls_closed_in_between_only_count_the_last_pull():
    seq = []
    for _ in range(3):
        seq += [pulled(0.05), window(0.0, hand="off", fingers=1.0)]
    seq += [pulled(0.06)]
    assert trial(*seq) == 0.25  # 6 cm: only the last pull is still open


def test_closing_part_way_takes_that_much_credit_back():
    assert trial(pulled(0.16), window(0.10, hand="off", fingers=1.0), window(0.16, hand="push", push=7.0)) == 0.5


def test_closing_and_reopening_cannot_count_more_than_it_is_open():
    assert trial(pulled(0.10), window(0.0, hand="off"), pulled(0.10)) == 0.5  # 20 cm pulled in all, but only 10 open


def test_success_needs_the_goal_the_handle_and_a_still_sash():
    RESET()
    TASK.score(window())
    assert TASK.success(pulled(0.18)) is True
    assert TASK.success(pulled(0.18, speed=0.05)) is False
    RESET()
    TASK.score(window())
    assert TASK.success(pulled(0.14)) is False
    RESET()
    TASK.score(window())
    assert TASK.success(pulled(0.18, torn=True)) is False


def test_distance_counts_down_to_the_goal():
    trial(pulled(0.0))
    assert TASK.distance(window(0.0)) == pytest.approx(T.GOAL)
    trial(pulled(0.10))
    assert TASK.distance(window(0.10)) == pytest.approx(T.GOAL - 0.10)
    trial(pulled(0.20))
    assert TASK.distance(window(0.20)) == 0.0


def test_the_tear_force_is_about_14_newtons():
    assert T.HANDLE_K * T.TEAR_AT == pytest.approx(14.4)
    assert T.HANDLE.breaks_at == T.TEAR_AT and T.HANDLE.low < -T.TEAR_AT  # it can tear before it hits its stop


def test_the_sash_can_be_freed_and_slid_without_tearing_the_handle():
    tear = T.HANDLE_K * T.TEAR_AT
    assert T.SLIDING_N < T.BREAKAWAY_N < tear  # a pull between 10 and 14 N frees it; sliding needs only 7 N
    assert tear - T.BREAKAWAY_N >= 4.0  # a usable margin between freeing the sash and tearing the handle
    assert T.SLIDING_N / T.HANDLE_K >= T.GIVE_MIN  # while it slides, the handle carries the pull (counts as pulled)


def test_the_overpull_variant_runs_into_the_end_stop():
    assert T.SASH_TRAVEL < 0.45


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
