"""Offline checks of the task file (no Isaac Sim): it loads, its seeded variants, its scoring rules and its demo's reach.

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
from yam_arena import station  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import load  # noqa: E402

import task as T  # noqa: E402
from fruitkit import Fruit, FruitState  # noqa: E402

RUN_LIMIT_S = 120.0  # 2,400 steps at 20 Hz
FAR = {"left_robot": {"tcp": np.array([0.1, 0.3, 1.0]), "gap": 0.09},
       "right_robot": {"tcp": np.array([0.1, -0.3, 1.0]), "gap": 0.09}}
BOWL = np.array([T.BOWL_XY[0], T.BOWL_XY[1], T.TOP + 0.03])
BAG = np.array([(T.BAG_X[0] + T.BAG_X[1]) / 2, 0.0, T.TOP + T.PAPER + T.SEMI[2]])


def kiwi(pos, speed=0.0, force=None, hands=None):
    p = np.asarray(pos, dtype=float)
    return FruitState(pos=p, quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.array([speed, 0, 0]), ang_vel=np.zeros(3),
                      low=p - 0.03, high=p + 0.03, force={} if force is None else force, hands=hands or FAR)


def start():
    return {n: kiwi((x, y, T.KZ)) for n, (x, y) in T.POCKETS.items()}


def sorted_right(ripe=None):
    ripe = T.RIPE if ripe is None else ripe
    return {n: kiwi(BOWL if n in ripe else BAG) for n in T.NAMES}


@pytest.fixture(autouse=True)
def fresh():
    T.reset_trial()
    yield
    T.reset_trial()


def test_loads():
    task = load(HERE / "task.py")
    assert set(task.free_objects) == set(T.NAMES)
    assert all(isinstance(o, Fruit) for o in task.objects if o.name in T.NAMES)


def test_variants():
    assert len(T.VARIANTS) == 6 and len(set(T.VARIANTS)) == 6
    assert all(len(v) == 2 for v in T.VARIANTS)
    for v in T.VARIANTS:
        ks = {k.name: k.stiffness for k in T.kiwis(frozenset(v))}
        assert sorted(ks.values()) == [T.RIPE_K, T.RIPE_K, T.HARD_K, T.HARD_K]
        # the ripe and hard kiwis are the same shape, size, mass and colour: only the squeeze tells them apart
        looks = {(k.semi, k.mass, k.color) for k in T.kiwis(frozenset(v))}
        assert len(looks) == 1


def test_squeeze_physics():
    # each pad driven 2 mm past first touch, through the finger drive (5000 N/m) in series with the contact spring
    assert -T.SQUEEZE == pytest.approx(T.SEMI[1] - T.PAD_INSET - 0.002)
    finger_k = 5000.0
    ripe_f = 0.002 / (1 / finger_k + 1 / T.RIPE_K)
    hard_f = 0.002 / (1 / finger_k + 1 / T.HARD_K)
    assert 4.0 < ripe_f < 6.0 and hard_f > 9.0
    assert ripe_f < T.CRUSH_N and 2 * ripe_f / T.RIPE_K > 0.0015  # gives about 2 mm in all, uncrushed
    assert 2 * hard_f / T.HARD_K < 0.0003  # a hard one gives under 0.3 mm
    assert 2 * T.CRUSH_N / T.RIPE_K == pytest.approx(0.006)  # the crush limit is a 6 mm squeeze
    # a full 50 N grip crushes a ripe kiwi; a gentle grip still lifts one (friction 0.9 on two pads)
    assert 50.0 > T.CRUSH_N and 2 * 0.9 * ripe_f > 3 * T.MASS * 9.81


def test_nothing_done_scores_zero():
    assert T.score(start()) == 0.0


def test_all_right():
    assert T.score(sorted_right()) == 1.0
    assert T.success(sorted_right())


def test_all_in_one_container_scores_zero():
    assert T.score({n: kiwi(BOWL) for n in T.NAMES}) == 0.0
    assert T.score({n: kiwi(BAG) for n in T.NAMES}) == 0.0


def test_partial_and_wrong():
    o = start()
    ripe = sorted(T.RIPE)
    hard = sorted(set(T.NAMES) - T.RIPE)
    o[ripe[0]] = kiwi(BOWL)
    assert T.score(o) == 0.25
    o[hard[0]] = kiwi(BAG)
    assert T.score(o) == 0.5
    o[hard[1]] = kiwi(BOWL)  # wrong: cancels one right
    assert T.score(o) == 0.25
    o[ripe[1]] = kiwi(BAG)  # wrong again
    assert T.score(o) == 0.0


def test_crushed_ripe_counts_nothing():
    ripe = sorted(T.RIPE)[0]
    o = start()
    o[ripe] = kiwi(o[ripe].pos, force={"left_robot/tip_left": T.CRUSH_N + 1})
    T.score(o)
    final = sorted_right()
    assert T.score(final) == 0.75  # it stays crushed once it has been
    # a hard kiwi squeezed as hard is fine
    T.reset_trial()
    hard = sorted(set(T.NAMES) - T.RIPE)[0]
    o = start()
    o[hard] = kiwi(o[hard].pos, force={"left_robot/tip_left": 50.0})
    T.score(o)
    assert T.score(sorted_right()) == 1.0


def test_held_or_moving_counts_nothing():
    o = sorted_right()
    n = T.NAMES[0]
    o[n] = kiwi(o[n].pos, force={"right_robot/tip_right": 3.0})
    assert T.score(o) == 0.75
    o[n] = kiwi(o[n].pos, speed=0.05)
    assert T.score(o) == 0.75
    # contacts unknown: a gripper's tool point near the kiwi counts as holding it
    hands = {"left_robot": {"tcp": o[n].pos + 0.01, "gap": 0.05}, "right_robot": FAR["right_robot"]}
    o[n] = FruitState(pos=o[n].pos, quat_xyzw=o[n].quat_xyzw, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=o[n].low,
                      high=o[n].high, force=None, hands=hands)
    assert T.score(o) == 0.75


def test_containers():
    assert T.in_bowl(kiwi(BOWL)) and not T.in_bag(kiwi(BOWL))
    assert T.in_bag(kiwi(BAG)) and not T.in_bowl(kiwi(BAG))
    on_rim = BOWL + np.array([0.0, 0.0, T.BOWL_RIM - T.TOP])
    assert not T.in_bowl(kiwi(on_rim))
    on_table = (T.BOWL_XY[0] + 0.13, 0.0, T.TOP + T.SEMI[2])
    assert not T.in_bowl(kiwi(on_table)) and not T.in_bag(kiwi(on_table))
    on_floor = (0.3, 0.0, 0.03)
    assert T.score({n: kiwi(on_floor) for n in T.NAMES}) == 0.0


@pytest.mark.parametrize("seed", range(6))
def test_demo_reaches_and_fits(seed, monkeypatch):
    monkeypatch.setattr(T, "RIPE", frozenset(T.VARIANTS[seed]))
    plans = {arm: ArmPlan(w, station.BASES[arm]) for arm, w in T.demo().items()}
    end = max(p.duration for p in plans.values())
    assert end < RUN_LIMIT_S / 3
    tcp = {}
    for arm, p in plans.items():
        pts = []
        for t in np.arange(0.0, end, 0.05):
            p.action7(t)
            pts.append(p.target(t)[0])
        assert p.worst_ik_mm < 2.0 and p.worst_ik_deg < 1.0, (arm, p.worst_ik_mm, p.worst_ik_deg)
        tcp[arm] = np.array(pts)
    gap = np.linalg.norm(tcp["left"][:, :2] - tcp["right"][:, :2], axis=1)
    assert gap.min() > 0.15  # the two grippers never come within 15 cm of each other (plan view)
    # every kiwi is squeezed: each arm's plan closes to the gentle squeeze over each of its kiwis
    for n, (x, y) in T.POCKETS.items():
        arm = "left" if y > 0 else "right"
        assert any(w.finger == T.SQUEEZE and abs(w.pos[0] - x) < 1e-9 and abs(w.pos[1] - y) < 1e-9
                   for w in T.demo()[arm])
