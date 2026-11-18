"""Offline checks of the task file (no Isaac Sim): it loads, its seeded variants, its scoring rules (the squeeze rule,
the crush rule, and the exact chances of guessing), and its demo's reach.

    YAM_ARENA=~/yam-isaac-lab-arena python -m pytest test_task.py
"""

from __future__ import annotations

import itertools
import os
import pathlib
import sys
from fractions import Fraction

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

RUN_LIMIT_S = 180.0  # 3,600 steps at 20 Hz
DT = 0.05  # one control step
FAR = {"left_robot": {"tcp": np.array([0.1, 0.3, 1.0]), "gap": 0.09},
       "right_robot": {"tcp": np.array([0.1, -0.3, 1.0]), "gap": 0.09}}
BOWL = np.array([T.BOWL_XY[0], T.BOWL_XY[1], T.TOP + 0.03])
BAG = np.array([(T.BAG_X[0] + T.BAG_X[1]) / 2, 0.0, T.TOP + T.PAPER + T.SEMI[2]])
GENTLE = {"left_robot/tip_left": 6.0, "left_robot/tip_right": 6.5}  # the demo's squeeze on a ripe kiwi
FULL = {"left_robot/tip_left": 50.0, "left_robot/tip_right": 50.0}  # the gripper's full force


def kiwi(pos, speed=0.0, force=None, hands=None, t=None):
    p = np.asarray(pos, dtype=float)
    return FruitState(pos=p, quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.array([speed, 0, 0]), ang_vel=np.zeros(3),
                      low=p - 0.03, high=p + 0.03, force={} if force is None else force, hands=hands or FAR, t=t)


def home(n):
    x, y = T.POCKETS[n]
    return (x, y, T.KZ)


def frame(t, where=None, force=None):
    """All eight kiwis at time t: in their pockets unless ``where`` says otherwise; ``force`` (kiwi -> contacts)."""
    where, force = where or {}, force or {}
    return {n: kiwi(where.get(n, home(n)), force=force.get(n), t=t) for n in T.NAMES}


class Trial:
    """Feeds score() a sequence of control steps, like the simulation does."""

    def __init__(self):
        T.reset_trial()
        self.t = 0.0
        self.last = T.score(frame(self.t))

    def step(self, where=None, force=None, seconds=DT):
        for _ in range(max(1, round(seconds / DT))):
            self.t += DT
            self.last = T.score(frame(self.t, where, force))
        return self.last

    def squeeze(self, names, how=GENTLE, seconds=1.0):
        return self.step(force={n: how for n in names}, seconds=seconds)


def place(ripe, bowl=(), bag=()):
    return {**{n: BOWL for n in bowl}, **{n: BAG for n in bag}}


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
    assert len(T.NAMES) == 8 and len(T.VARIANTS) == 70 and len(set(T.VARIANTS)) == 70
    assert all(len(v) == 4 for v in T.VARIANTS)
    for v in T.VARIANTS:
        ks = {k.name: k.stiffness for k in T.kiwis(frozenset(v))}
        assert sorted(ks.values()) == [T.RIPE_K] * 4 + [T.HARD_K] * 4
        # the ripe and hard kiwis are the same shape, size, mass and colour: only the squeeze tells them apart
        looks = {(k.semi, k.mass, k.color) for k in T.kiwis(frozenset(v))}
        assert len(looks) == 1
    # every pocket is ripe in exactly half the layouts
    for n in T.NAMES:
        assert sum(n in v for v in T.VARIANTS) == 35


def test_squeeze_physics():
    # each pad driven 2 mm past first touch, through the finger drive (5000 N/m) in series with the contact spring
    assert -T.SQUEEZE == pytest.approx(T.SEMI[1] - T.PAD_INSET - 0.002)
    finger_k = 5000.0
    ripe_f = 0.002 / (1 / finger_k + 1 / T.RIPE_K)
    hard_f = 0.002 / (1 / finger_k + 1 / T.HARD_K)
    assert 4.0 < ripe_f < 6.0 and hard_f > 9.0
    assert T.SQUEEZE_N < ripe_f < T.CRUSH_N and 2 * ripe_f / T.RIPE_K > 0.0015  # gives about 2 mm in all, uncrushed
    assert 2 * hard_f / T.HARD_K < 0.0003  # a hard one gives under 0.3 mm
    assert 2 * T.CRUSH_N / T.RIPE_K == pytest.approx(0.006)  # the crush limit is a 6 mm squeeze
    # a full 50 N grip crushes a ripe kiwi; a gentle grip still lifts one (friction 0.9 on two pads)
    assert 50.0 > T.CRUSH_N and 2 * 0.9 * ripe_f > 3 * T.MASS * 9.81


def test_net_to_score():
    assert [T.net_to_score(n) for n in range(-8, 9)] == [0.0] * 13 + [0.25, 0.5, 0.75, 1.0]


def test_nothing_done_scores_zero():
    tr = Trial()
    assert tr.step(seconds=100) == 0.0


def test_all_squeezed_and_sorted_scores_one():
    tr = Trial()
    tr.squeeze(T.NAMES)
    ripe, hard = sorted(T.RIPE), sorted(set(T.NAMES) - T.RIPE)
    assert tr.step(place(T.RIPE, ripe, hard)) == 1.0
    assert T.success(frame(tr.t, place(T.RIPE, ripe, hard)))


def test_sorting_without_squeezing_scores_zero():
    # every kiwi in its right container, but none was squeezed in its pocket first: nothing counts
    tr = Trial()
    ripe, hard = sorted(T.RIPE), sorted(set(T.NAMES) - T.RIPE)
    assert tr.step(place(T.RIPE, ripe, hard), seconds=5) == 0.0
    # and an unsqueezed kiwi in the wrong container still counts against
    tr = Trial()
    tr.squeeze(T.NAMES[:7])
    w = place(T.RIPE, ripe, hard)
    odd = T.NAMES[7]
    w[odd] = BAG if odd in T.RIPE else BOWL
    assert tr.step(w) == 0.5  # 7 right - 1 wrong = net 6


def test_squeeze_rule():
    n = sorted(set(T.NAMES) - T.RIPE)[0]  # a hard kiwi, to the bag
    tr = Trial()
    tr.squeeze([n], seconds=0.4)  # too short
    assert n not in T._squeezed
    tr.step(seconds=1.0)
    tr.squeeze([n], seconds=0.15)  # 0.55 s in all, in two goes: counts
    assert n in T._squeezed
    # one pad only, or one pad from each gripper, is not a squeeze
    m = sorted(set(T.NAMES) - T.RIPE)[1]
    tr.squeeze([m], how={"left_robot/tip_left": 8.0}, seconds=2.0)
    tr.squeeze([m], how={"left_robot/tip_left": 8.0, "right_robot/tip_right": 8.0}, seconds=2.0)
    assert m not in T._squeezed
    # pads touching under 1 N each: not a squeeze
    tr.squeeze([m], how={"left_robot/tip_left": 0.5, "left_robot/tip_right": 0.5}, seconds=2.0)
    assert m not in T._squeezed
    # squeezed only after it left its pocket (lifted 2 cm, or slid 3 cm): doesn't count
    k = sorted(set(T.NAMES) - T.RIPE)[2]
    x, y, z = home(k)
    for where in ((x, y, z + 0.02), (x + 0.03, y, z)):
        tr.step(where={k: where}, force={k: GENTLE}, seconds=2.0)
    assert k not in T._squeezed
    # the right arm's pads count as well as the left's
    tr.squeeze([k], how={"right_robot/tip_left": 5.0, "right_robot/tip_right": 5.0}, seconds=0.6)
    assert k in T._squeezed


def test_clock_unknown_falls_back_to_counting_calls():
    T.reset_trial()
    n = T.NAMES[0]
    for _ in range(round(T.SQUEEZE_S / T.FALLBACK_DT) + 1):
        T.score({m: kiwi(home(m), force=GENTLE if m == n else None) for m in T.NAMES})
    assert n in T._squeezed


def test_full_force_squeezing_scores_zero():
    # squeezing every kiwi with the gripper's full force tells them apart, but crushes the ripe ones: those count for
    # nothing in the bowl, so at best the four hard ones are right: net 4, score 0
    tr = Trial()
    tr.squeeze(T.NAMES, how=FULL)
    assert T._crushed == set(T.RIPE)
    ripe, hard = sorted(T.RIPE), sorted(set(T.NAMES) - T.RIPE)
    assert tr.step(place(T.RIPE, ripe, hard)) == 0.0
    # a hard kiwi takes a full grip unharmed
    tr = Trial()
    tr.squeeze(hard, how=FULL)
    tr.squeeze(ripe)
    assert tr.step(place(T.RIPE, ripe, hard)) == 1.0


def test_partial_and_wrong():
    ripe, hard = sorted(T.RIPE), sorted(set(T.NAMES) - T.RIPE)
    tr = Trial()
    tr.squeeze(T.NAMES)
    # four right and nothing else: net 4, nothing yet
    assert tr.step(place(T.RIPE, ripe[:2], hard[:2])) == 0.0
    assert tr.step(place(T.RIPE, ripe[:3], hard[:2])) == 0.25
    assert tr.step(place(T.RIPE, ripe[:3], hard[:3])) == 0.5
    assert tr.step(place(T.RIPE, ripe, hard[:3])) == 0.75
    # all eight placed, one swapped pair (two wrong): net 4
    assert tr.step(place(T.RIPE, ripe[:3] + hard[:1], hard[1:] + ripe[3:])) == 0.0
    # all eight placed, one kiwi wrong: net 6
    assert tr.step(place(T.RIPE, ripe + hard[:1], hard[1:])) == 0.5
    # all in one container: net 0
    assert tr.step(place(T.RIPE, T.NAMES, ())) == 0.0
    assert tr.step(place(T.RIPE, (), T.NAMES)) == 0.0


def test_held_or_moving_counts_nothing():
    ripe, hard = sorted(T.RIPE), sorted(set(T.NAMES) - T.RIPE)
    tr = Trial()
    tr.squeeze(T.NAMES)
    w = place(T.RIPE, ripe, hard)
    n = hard[0]
    assert tr.step(w, force={n: {"right_robot/tip_right": 3.0}}) == 0.75  # still held
    T._last_t[0] = None
    o = frame(tr.t + DT, w)
    o[n] = kiwi(o[n].pos, speed=0.05, t=tr.t + DT)  # still rolling
    assert T.score(o) == 0.75
    # contacts unknown: a gripper's tool point near the kiwi counts as holding it
    hands = {"left_robot": {"tcp": o[n].pos + 0.01, "gap": 0.05}, "right_robot": FAR["right_robot"]}
    o[n] = FruitState(pos=o[n].pos, quat_xyzw=o[n].quat_xyzw, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=o[n].low,
                      high=o[n].high, force=None, hands=hands, t=tr.t + 2 * DT)
    assert T.score(o) == 0.75


def test_containers():
    assert T.in_bowl(kiwi(BOWL)) and not T.in_bag(kiwi(BOWL))
    assert T.in_bag(kiwi(BAG)) and not T.in_bowl(kiwi(BAG))
    on_rim = BOWL + np.array([0.0, 0.0, T.BOWL_RIM - T.TOP])
    assert not T.in_bowl(kiwi(on_rim))
    on_table = (T.BOWL_XY[0] + 0.15, 0.0, T.TOP + T.SEMI[2])
    assert not T.in_bowl(kiwi(on_table)) and not T.in_bag(kiwi(on_table))
    # the containers clear the trays and each other (plan view)
    tray_in = T.COL_Y[0] - T.POCKET_IN[1] / 2 - T.WALL_T
    assert T.BAG_Y[1] + T.PAPER + 0.008 < tray_in
    # the gripper's body (wider than its fingers) clears the bag's rolled top over the near pockets
    assert T.ROW_X[0] - (T.BAG_X[1] + T.PAPER + 0.008) >= 0.06 - 1e-9
    tray_far = T.ROW_X[1] + T.POCKET_IN[0] / 2 + T.WALL_T
    assert T.BOWL_XY[0] - T.BOWL_RIM_R > T.BAG_X[1] + 0.05
    assert math_hypot_clear(tray_far, tray_in)


def math_hypot_clear(tray_far, tray_in):
    """The bowl's rim stays clear of the trays' inner far corners."""
    import math
    return math.hypot(T.BOWL_XY[0] - tray_far, tray_in) > T.BOWL_RIM_R


# -- guessing: the exact chances --------------------------------------------------------------------------------------

def blind_distribution(n_bowl, n_bag):
    """Score distribution (exact fractions) of a robot that squeezes every kiwi gently but can't tell them apart, and
    puts ``n_bowl`` of them in the bowl and ``n_bag`` in the bag: over the 70 equally likely layouts. (The layouts are
    exchangeable, so which pockets it picks doesn't matter.)"""
    bowl, bag = T.NAMES[:n_bowl], T.NAMES[n_bowl:n_bowl + n_bag]
    dist = {}
    for v in T.VARIANTS:
        T.RIPE = frozenset(v)
        tr = Trial()
        tr.squeeze(T.NAMES, seconds=0.6)
        s = Fraction(tr.step(place(T.RIPE, bowl, bag))).limit_denominator(4)
        dist[s] = dist.get(s, 0) + Fraction(1, len(T.VARIANTS))
    return dist


@pytest.fixture
def keep_ripe():
    ripe = T.RIPE
    yield
    T.RIPE = ripe


def test_guessing_chances(keep_ripe):
    worst_mean, worst_full, worst_any = Fraction(0), Fraction(0), Fraction(0)
    for b in range(9):
        for g in range(9 - b):
            d = blind_distribution(b, g)
            assert sum(d.values()) == 1
            mean = sum(k * p for k, p in d.items())
            worst_mean = max(worst_mean, mean)
            worst_full = max(worst_full, d.get(Fraction(1), Fraction(0)))
            worst_any = max(worst_any, sum(p for k, p in d.items() if k > 0))
            if (b, g) == (4, 4):  # four and four: full score 1 in 70, else 0
                assert d == {Fraction(1): Fraction(1, 70), Fraction(0): Fraction(69, 70)}
            if (b, g) == (5, 3):  # the best blind split: 0.5 with chance 1 in 14
                assert d == {Fraction(1, 2): Fraction(1, 14), Fraction(0): Fraction(13, 14)}
    assert worst_mean == Fraction(1, 28)  # 0.036: no blind strategy averages more
    assert worst_full == Fraction(1, 70)  # 1.4%
    assert worst_any == Fraction(1, 14)  # 7.1%


def test_coin_flip_guessing_exact():
    # a coin flip for each kiwi (ignoring that four are ripe): right count ~ Binomial(8, 1/2), net = 2 * right - 8
    dist = {}
    for right in range(9):
        s = Fraction(T.net_to_score(2 * right - 8)).limit_denominator(4)
        p = Fraction(len(list(itertools.combinations(range(8), right))), 256)
        dist[s] = dist.get(s, 0) + p
    assert dist == {Fraction(0): Fraction(247, 256), Fraction(1, 2): Fraction(8, 256), Fraction(1): Fraction(1, 256)}
    assert sum(k * p for k, p in dist.items()) == Fraction(5, 256)  # 0.020


# -- the demo -----------------------------------------------------------------------------------------------------

def seeds_covering_each_arms_patterns():
    """A few seeds that, between them, give each arm every one of its 16 ripe/hard patterns."""
    want = {(side, pat) for side in "LR" for pat in itertools.product((0, 1), repeat=4)}
    seeds = []
    for s, v in enumerate(T.VARIANTS):
        got = {(side, tuple(int(f"kiwi_{side}{i}" in v) for i in "1234")) for side in "LR"}
        if got & want:
            seeds.append(s)
            want -= got
        if not want:
            return seeds
    raise AssertionError(f"patterns never drawn: {want}")


def test_demo_reaches_and_fits(keep_ripe):
    for seed in seeds_covering_each_arms_patterns():
        T.RIPE = frozenset(T.VARIANTS[seed])
        plans = {arm: ArmPlan(w, station.BASES[arm]) for arm, w in T.demo().items()}
        end = max(p.duration for p in plans.values())
        assert end < RUN_LIMIT_S / 3
        tcp = {}
        for arm, p in plans.items():
            pts = []
            for t in np.arange(0.0, end, 0.1):
                p.action7(t)
                pts.append(p.target(t)[0])
            assert p.worst_ik_mm < 2.0 and p.worst_ik_deg < 1.0, (seed, arm, p.worst_ik_mm, p.worst_ik_deg)
            tcp[arm] = np.array(pts)
        # wherever a gripper's open fingers could be over the bag's rolled top, the pad tips stay above it
        for arm, pts in tcp.items():
            over = ((pts[:, 0] > T.BAG_X[0] - 0.03) & (pts[:, 0] < T.BAG_X[1] + 0.03)
                    & (np.abs(pts[:, 1]) < T.BAG_Y[1] + 0.06))
            tips = pts[over, 2] - 0.013
            assert tips.size == 0 or tips.min() > T.TOP + T.BAG_H + 0.005, (seed, arm, tips.min() - T.TOP)
        gap = np.linalg.norm(tcp["left"][:, :2] - tcp["right"][:, :2], axis=1)
        assert gap.min() > 0.15, (seed, gap.min())  # the two grippers never come within 15 cm (plan view)
        # every kiwi is squeezed in its pocket and held there for a second before the lift
        for n, (x, y) in T.POCKETS.items():
            w = T.demo()["left" if y > 0 else "right"]
            held = [a for a, b in zip(w, w[1:]) if a.finger == T.SQUEEZE and b.finger == T.SQUEEZE
                    and a.pos == b.pos and abs(a.pos[0] - x) < 1e-9 and abs(a.pos[1] - y) < 1e-9]
            assert held and held[0].pos[2] == T.GRIP_Z
