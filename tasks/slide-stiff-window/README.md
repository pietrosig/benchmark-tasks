# Slide open a small stiff window by its delicate handle

A small sliding window stands on the table, fixed, facing the arms. Its left half is a sash that slides right to open;
it is stiff, dragging harder the faster it moves (about 3 N for every cm/s). Its only handle is a slim chrome D-handle
that tears off if pulled with more than about 14 N. The robot has to open the sash 15 cm, slowly enough that the handle
holds. In the runs below, pulls of 4 s or longer opened it with the handle on, and 3 s and 2 s pulls tore the handle
off after 3 to 5 cm. Only the opening made by pulling the gripped handle counts, so pushing the sash open directly scores
nothing. A robot that does nothing scores 0, and so does one that tears the handle off, however far the window opens.

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
for good and it slides freely. Each step the kit also reports where each gripper's tool point is and how far its fingers
are open, so the score can tell whether the sash is being pulled by its handle.

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
video): the sash 17.9 cm open, 16.1 cm of it pulled by the gripped handle (the last 1.8 cm came while the fingers
opened, so it doesn't count), the handle never more than 7.5 mm from rest (about 6 N) (`outcome: succeeded`,
`task_score: 1`), well within the 1,200-step limit. While the fingers are closed on the bar they read 0.13 open.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Pulled by its handle:** a gripper's tool point is within 1 cm of the grip bar's centre (in depth and along the
  window, and within the bar's height), its fingers are closed on the bar (opening 0.05 to 0.25 of fully open: closed on the
  bar it reads 0.13; open around it 0.38 and shut empty 0 don't count), and the handle's fixings have given at least 1 mm (0.8 N) in the opening direction.
- **Pulled open** is the sash travel (opening) made while it is pulled by its handle, added up over the trial, and never
  more than how far it is open at that moment. Pushing the sash or its frame, pushing the handle without gripping it, a
  gripped handle that carries no pull (another hand pushing the sash), or the sash moving after the gripper has let go
  adds nothing. In the real setup: from the trial video, the opening made between each time a gripper closes on the grip
  bar and lets go, while no other part of the robot touches the sash, its glass or the frame, measured with a ruler from
  the left jamb to the sash's left edge.
- **Torn:** the handle has given 18 mm along the sash, which takes about 14 N, at any point in the trial. It stays torn.
- With the handle on: **1.0** at 15 cm or more pulled open (`success()` also needs the sash moving under 1 cm/s for 1 s,
  which ends the trial), **0.75** from 12 cm, **0.5** from 8 cm, **0.25** from 4 cm, **0** under 4 cm.
- **0** if the handle is torn, however far the sash is open.

## Real setup

The frame and sash are white uPVC (PVC-U) profile with 3 mm float glass, and the handle a chrome-plated zinc D-handle.
The sash's stiffness comes from a linear viscous damper under the sill, for example an adjustable air dashpot or a rotary
oil damper driving a pinion on a rack fixed to the sash's bottom rail, set to 300 N s/m: a steady 6 N pull on the
handle, read on a spring scale, moves the sash at 2 cm/s. Check it before each session. The handle is held on by a
breakaway fixing (a magnet or a calibrated shear pin) that lets go at about 14 N.

## Rushed pulls

Isaac runs of `task.py` with only the demo's pull length changed in a scratch wrapper (`task.PULL_S`), one run each.
The pull covers 21.2 cm; the rest of the demo is unchanged. These runs were made before the rule that only travel pulled
by the gripped handle counts, so the table shows how far the sash opened and whether the handle held, not a score;
the torn ones score 0 under either rule. The demo was rerun with the rule (above).

| Pull | Opened | Handle |
|---|---|---|
| Demo, 10 s | 17.9 cm | on (gave 7.5 mm at most) |
| 5 s | 16.8 cm | on (14.8 mm) |
| 4 s | 16.3 cm | on (17.7 mm, just under the 18 mm tear) |
| 3 s | 5.4 cm | torn off |
| 2 s | 2.9 cm | torn off |

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
