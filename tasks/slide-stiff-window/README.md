# Slide open a small stiff window by its delicate handle

A small sliding window stands on the table, fixed, facing the arms. Its left half is a sash that slides right to open;
it is stiff, dragging harder the faster it moves (about 3 N for every cm/s). Its only handle is a slim chrome D-handle
that tears off if pulled with more than about 14 N. The robot has to open the sash 15 cm, slowly enough that the handle
holds. In the runs below, pulls of 4 s or longer opened it with the handle on, and 3 s and 2 s pulls tore the handle
off after 3 to 5 cm. A robot that does nothing scores 0, and so does one that tears the handle off, however far the window
opens.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the window, `success()`, `score()`, `distance()` and `demo()` |
| `slidekit.py` | A fixture with sliding parts (prismatic joints with springs, damping and a break point) for yam_arena task files |
| `test_task.py` | Offline checks: the file loads, the scoring rules, the geometry, the demo's reach and joint limits |
| `preview.py` | Offline picture of the layout and the demo (top, side and front views), no GPU needed |

`slidekit.py` hooks `yam_arena.scene` when it is imported, without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. The window is one articulation fixed to the table: the
frame, the sash on a damped prismatic joint (300 N s/m), and the handle on a second prismatic joint on the sash, a
800 N/m spring that stands in for its fixings. Once the handle has given 18 mm (about 14 N) it is torn: its spring goes limp
for good and it slides freely.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The window is stiff and its handle is delicate. Grip the handle and slide the window open." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/slide-stiff-window-dream
```

`task_score` is 1.0 when the trial ends in success, otherwise `score()` at the end of the episode.

One Dream-1 (`dreamlabs/dream-1-beta`) trial with this command scored **0**: it ran the full 1,200 steps without
opening the window. The right arm reached to the window by the handle and stayed there; the sash never moved (0.2 mm at
most) and the handle was never pulled (0.1 mm of give at most). It made 25 requests in one Dream session, each on the
first attempt, and took about 3 minutes.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The window is stiff and its handle is delicate. Grip the handle and slide the window open." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/slide-stiff-window-demo
```

Open loop, right arm only, in one continuous motion after the first move from the park pose: it reaches round to the
handle from the room side, grips it, pulls the sash 20 cm open over 10 s (about 2 cm/s), opens its fingers over the
last 1.5 cm while still moving and draws back. With this command the demo succeeds after about 369 steps (18.45 s of
video): the sash 17.9 cm open, the handle never more than 7.5 mm from rest (about 6 N) (`outcome: succeeded`,
`task_score: 1`), well within the 1,200-step limit.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Open** is how far the sash has slid from closed (in the real setup: from the left jamb to the sash's left edge).
- **Torn:** the handle has given 18 mm along the sash, which takes about 14 N, at any point in the trial. It stays torn.
- With the handle on: **1.0** at 15 cm or more (`success()` also needs the sash moving under 1 cm/s for 1 s, which
  ends the trial), **0.75** from 12 cm, **0.5** from 8 cm, **0.25** from 4 cm, **0** under 4 cm.
- **0** if the handle is torn, however far the sash is open.
- How the sash is moved isn't scored; it may still be held at the end.

## Rushed pulls

Isaac runs of `task.py` with only the demo's pull length changed in a scratch wrapper (`task.PULL_S`), one run each.
The pull covers 21.2 cm; the rest of the demo is unchanged.

| Pull | Opened | Handle | Score |
|---|---|---|---|
| Demo, 10 s | 17.9 cm | on (gave 7.5 mm at most) | 1.0 |
| 5 s | 16.8 cm | on (14.8 mm) | 1.0 |
| 4 s | 16.3 cm | on (17.7 mm, just under the 18 mm tear) | 1.0 |
| 3 s | 5.4 cm | torn off | 0 |
| 2 s | 2.9 cm | torn off | 0 |

The give is sampled every 10 scorer calls, so true peaks may be slightly higher. The faster pulls that hold open the
sash a little less because the hand lags further behind its target against the drag.

## Runtime and seeds

- Run limit 1,200 steps (1 minute at 20 Hz).
- Wall time per trial on one RTX GPU, including Isaac Sim start-up: about 3 minutes for the scripted demo (about 4 with
  three other trials running on the same GPU).
- The layout is fixed with `-E object_jitter=0`, so every trial starts the same. There are no seeds.
- Set `WINDOW_TRACE=/some/file.jsonl` to log the sash's opening, the handle's give and the sash's speed every 10
  scorer calls (for debugging; it doesn't change the trial).

## Offline checks (no GPU)

```bash
git clone https://github.com/kairosiann/yam-isaac-lab-arena ~/yam-isaac-lab-arena
uv run --python 3.12 --with numpy --with matplotlib --with pytest python -m pytest test_task.py
uv run --python 3.12 --with numpy --with matplotlib python preview.py ~/slide-stiff-window-preview.png
```
