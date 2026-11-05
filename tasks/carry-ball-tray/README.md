# Carry an open tray of loose steel balls to a fixture

A shallow open tray stands on the table between the arms, with six loose 2 cm steel balls in it and only a 2 mm lip to
keep them there. The robot has to lift the tray by its two end lugs, carry it 17 cm and set it down in a raised
fixture without spilling the balls. In the runs below, a careful two-handed carry kept all six, hurried or tilted ones
lost balls, and a scripted carry by one arm tipped the tray and spilled them all. A robot that does nothing scores 0,
and so does one that doesn't get the tray into the fixture, whose 3 cm plate stands in the way of pushing it there.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()`, `score()`, `distance()` and `demo()` |
| `traykit.py` | One loose rigid object made of several boxes (the tray with its lip and lugs) for yam_arena task files |
| `test_task.py` | Offline checks: the file loads, the scoring rules, the geometry, the demo's reach and joint limits |
| `preview.py` | Offline picture of the layout and the demo (top, side and front views), no GPU needed |

`traykit.py` hooks `yam_arena.scene` when it is imported, without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment and the `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the tray in the fixture without spilling any of the steel balls." --scorer task_score \
  --max-steps 1200 --store-frames --log-dir ~/yam-eval/carry-ball-tray-dream
```

`task_score` is 1.0 when the trial ends in success, otherwise `score()` at the end of the episode.

One Dream-1 (`dreamlabs/dream-1-beta`) trial with this command scored **0**: it ran the full 1,200 steps
without getting the tray into the fixture. It made 25 requests; at step 128 the client got a new Dream session
and retried once by itself (`attempts: 2` in the log), and the trial carried on to the end. The trial ran before
the seating rule was relaxed (it then also required the tray to be let go, flat within 0.5 degrees); the scene, physics
and instruction are unchanged, and the tray never reached the fixture, so it scores 0 under either rule.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the tray in the fixture without spilling any of the steel balls." --scorer task_score \
  --max-steps 1200 --store-frames --log-dir ~/yam-eval/carry-ball-tray-demo
```

Open loop. Both arms grip a lug each, lift the tray over 3 s, carry it to the fixture over 5 s, pause, lower it over
3 s and let go. With this command the demo succeeds after about 332 steps (16.6 s of video) with all six balls in the
tray (`outcome: succeeded`, `task_score: 1`), well within the 1,200-step limit.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- The tray is **seated** when it lies in the pocket: centre within 1 cm of the pocket's, underside within 3 mm of the
  pocket's floor. A tray caught on a 1 cm rail is about 5 mm up and doesn't count. It may still be held: while the
  arms hold it in the pocket it is about 1.4 mm up.
- A ball is **in the tray** when it lies on the tray's floor inside the lip, not held up above it.
- With the tray seated: **1.0** with six balls in it (`success()` also needs everything moving under 2 cm/s for 1 s,
  which ends the trial), **0.75** with five, **0.5** with three or four, **0.25** with one or two.
- **0** with no balls in it, or if the tray isn't seated.

## Careless carries

Isaac runs of `task.py` with the demo changed in a scratch wrapper, one run each, scored with an earlier, stricter
seating rule (underside within 1 mm, tilted at most 0.5 degrees, let go). They show the balls really do spill,
and how much care the task asks for. Only the carry changes: the lift and the lowering keep their demo timing.

| Carry | Balls left | Tray | Score |
|---|---|---|---|
| Demo, 5 s | 6 | Seated | 1.0 |
| 0.8 s | 5 | Seated | 0.75 |
| 5 s with the left arm 8 cm higher (up to 15 degrees of tilt) | 5 | Seated | 0.75 |
| 0.3 s | 0 | Not seated | 0 |
| Left arm only | 0 | Back on the table near its start, up to 31 degrees of tilt on the way | 0 |

## Runtime and seeds

- Run limit 1,200 steps (1 minute at 20 Hz).
- Wall time per trial on one RTX GPU, including Isaac Sim start-up: about 3 minutes for the scripted demo (about 4 with
  four other trials running on the same GPU). The Dream-1 trial took about 4 minutes.
- The layout is fixed with `-E object_jitter=0`, so every trial starts the same. There are no seeds.

## Offline checks (no GPU)

```bash
git clone https://github.com/kairosiann/yam-isaac-lab-arena ~/yam-isaac-lab-arena
uv run --python 3.12 --with numpy --with matplotlib --with pytest python -m pytest test_task.py
uv run --python 3.12 --with numpy --with matplotlib python preview.py ~/carry-ball-tray-preview.png
```
