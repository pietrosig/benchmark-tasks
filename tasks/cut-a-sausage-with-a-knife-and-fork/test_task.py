"""Offline checks (no simulator): layout, scoring logic and the demo's reach."""
from __future__ import annotations

import importlib.util
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("task", HERE / "task.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


class S:  # a stand-in object state
    def __init__(self, low, high):
        self.low, self.high = np.asarray(low, float), np.asarray(high, float)


def _set(through=(), pinned=None, touched=False, off=False, pierced=None, open_=(1.0, 1.0), tcps=None):
    T._reset()
    for k in through:
        T.EP["through"][k] = True
        T.EP["pinned_cut"][k] = True if pinned is None else (k in pinned)
    T.EP["touched"], T.EP["off"], T.EP["pierced"] = (["left/finger"] if touched else []), off, pierced
    T.EP["open"] = list(open_)
    T.EP["tcps"] = tcps or [np.array([0.1, 0.3, 1.0]), np.array([0.1, -0.3, 1.0])]


def _laid(x, y):
    return S((x - 0.09, y - 0.01, T.PF), (x + 0.09, y + 0.01, T.PF + 0.012))


OBJ = {"fork": _laid(0.30, 0.08), "knife": _laid(0.30, -0.08)}


def test_layout():
    assert len(T.SLABS) == T.N == 13 and abs(T.N * T.SL - 0.13) < 1e-9
    assert abs(sum(s.mass for s in T.SLABS) - T.MASS) < 1e-9
    assert T.SLABS[0].pos[1] < T.SLABS[-1].pos[1]


def test_nothing_done_scores_zero():
    _set()
    assert T.score(OBJ) == 0.0


def test_three_pieces_and_laid_utensils_score_one():
    _set(through=(2, 4, 6))
    assert len(T._valid_pieces()) == 3
    assert T.score(OBJ) == 1.0


def test_pieces_need_the_fork_in():
    _set(through=(2, 4, 6), pinned={2, 4})
    assert T.score(OBJ) == 0.5


def test_piece_sizes():
    _set(through=(1,))  # 1 cm: too small
    assert T.score(OBJ) == 0.0
    _set(through=(3,))  # 3 cm: too big
    assert T.score(OBJ) == 0.0
    _set(through=(2, 5, 7))  # 2, 3 and 2 cm
    assert T.score(OBJ) == 0.5
    _set(through=(2, 11))  # one from each end
    assert T.score(OBJ) == 0.5


def test_touch_or_off_plate_latches_zero():
    _set(through=(2, 4, 6), touched=True)
    assert T.score(OBJ) == 0.0
    _set(through=(2, 4, 6), off=True)
    assert T.score(OBJ) == 0.0


def test_utensils_must_be_laid_down_and_let_go():
    _set(through=(2, 4, 6), pierced=10)  # fork left in the food
    assert T.score(OBJ) == 0.75
    _set(through=(2, 4, 6), open_=(0.1, 1.0))  # still gripping
    assert T.score(OBJ) == 0.75
    on_table = dict(OBJ, knife=_laid(0.30, -0.30))
    _set(through=(2, 4, 6))
    assert T.score(on_table) == 0.75
    upright = dict(OBJ, fork=S((0.28, 0.0, T.PF), (0.32, 0.03, T.PF + 0.12)))
    assert T.score(upright) == 0.75
    near = [np.array([0.30, 0.08, T.PF + 0.02]), np.array([0.1, -0.3, 1.0])]
    _set(through=(2, 4, 6), tcps=near)
    assert T.score(OBJ) == 0.75


def test_demo_reaches():
    from yam_arena import station
    from yam_arena.demo import ArmPlan

    plans = T.demo()
    for arm, w in plans.items():
        p = ArmPlan(list(w), station.BASES[arm])
        t = 0.0
        while t < p.duration:
            p.action7(t)
            t += 0.1
        assert p.worst_ik_mm < 5.0 and p.worst_ik_deg < 2.0, (arm, p.worst_ik_mm, p.worst_ik_deg)
    assert max(sum(x.duration for x in w) for w in plans.values()) * 20 < 2400
