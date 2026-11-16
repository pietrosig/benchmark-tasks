# Slide open a small stiff window by its delicate handle

A small sliding window stands on the table, fixed, facing the arms. Its left half is a sash that slides right to open.
It is stiff in its track, like a window whose runners have dried out: it stays put until pulled with 10 N, then slides
with a steady 7 N drag, at any speed, until it stops at 24 cm open. Its only handle is a slim chrome D-handle that tears
off if pulled with more than about 14 N. The robot has to pull the sash 15 cm open by its handle with a force between
what frees the sash and what tears the handle, and stop pulling when the sash reaches its stop: in the runs below, the
demo opens it with the handle on, and a pull that carries on past the stop tears the handle off. Only the opening made
by pulling the gripped handle counts, so pushing the sash open directly scores nothing. A robot that does nothing scores
0, and so does one that tears the handle off, however far the window opens.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the window, `success()`, `score()`, `distance()` and `demo()` |
| `slidekit.py` | A fixture with sliding parts (prismatic joints with springs, dry friction and a break point) for yam_arena task files |
| `test_task.py` | Offline checks: the file loads, the scoring rules, the geometry, the demo's reach and joint limits |
| `preview.py` | Offline picture of the layout and the demo (top, side and front views), no GPU needed |

`slidekit.py` hooks `yam_arena.scene` when it is imported, without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. The window is one articulation fixed to the table: the
frame, the sash on a prismatic joint with dry friction applied by PhysX itself every physics step (a static friction
effort of 10 N and a dynamic one of 7 N: it holds until pulled with 10 N, then drags with 7 N at any speed; the sash
clears the sill and head by 2 mm, so nothing else drags on it), and the handle on a second prismatic joint on the sash, a 800 N/m spring that stands in for its fixings. Once the handle has given 18 mm (about 14 N) it is torn: its spring goes limp
for good and it slides freely. Each step the kit also reports where each gripper's tool point is and how far its fingers
are open, so the score can tell whether the sash is being pulled by its handle.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "The window is stiff and its handle is delicate. Grip the handle and slide the window open." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/slide-stiff-window-dream
```

`task_score` is 1.0 when the trial ends in success, otherwise `score()` at the end of the episode.

One Dream-1 (`dreamlabs/dream-1-beta`) trial with this command scored **0**: it ran the full 1,200 steps without
opening the window. Both arms moved to the window: the left arm (on the left of the video) hovered its gripper near
the handle (within 5 cm of the grip bar) and the right arm (on the right of the video) went to the window's right end;
near the end both drew back. Neither gripper ever
closed on the handle; the sash never moved (1.6 mm at most) and the handle was never pulled (0.1 mm of give at most).

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "The window is stiff and its handle is delicate. Grip the handle and slide the window open." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/slide-stiff-window-demo
```

Open loop, right arm only, in one continuous motion after the first move from the park pose: it reaches round to the
handle from the room side, grips it, pulls along the track over 10 s, aiming 23 cm (just short of the 24 cm stop),
opens its fingers over the last 1.5 cm while still moving and draws back. The hand leads the sash: the handle gives
until the pull reaches 10 N and frees the sash; at this slow pace it then judders open (stick-slip, as a stiff window
does), the handle carrying 6 to 10 N. With this command the demo succeeds after about 335 steps (16.8 s of video): the
sash 18.0 cm open, 17.9 cm of it pulled by the gripped handle, the handle never more than 12.5 mm from rest (10 N, under
the 14 N tear) (`outcome: succeeded`, `task_score: 1`), well within the 1,200-step limit. While the fingers are closed
on the bar they read 0.13 open.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Pulled by its handle:** a gripper's tool point is within 1 cm of the grip bar's centre (in depth and along the
  window, and within the bar's height), its fingers are closed on the bar (opening 0.05 to 0.25 of fully open: closed on the
  bar it reads 0.13; open around it 0.38 and shut empty 0 don't count), the handle's fixings have given at least 1 mm
  (0.8 N) in the opening direction, nothing else pushes the sash open (from the sash's force balance, its mass times its
  acceleration less the track's friction and the handle's pull, no other force helps it open by more than 2 N), and no
  other gripper is at the window (within 2 cm of the sash, its glass or the frame), even without pushing. So a light
  pull on the handle while another gripper, an arm or anything else pushes or rests on the sash earns nothing.
- **Pulled open** is a running total: the opening made while the sash is pulled by its handle alone is added, and every
  bit of closing, however it happens, is taken off. It is never more than how far the sash is open. Pushing the sash or
  its frame, pushing the handle without gripping it, a light pull while something else pushes, or the sash moving after
  the gripper has let go adds nothing, and pulling a little, closing and pulling again earns no more than the last pull
  (`test_task.py` checks both). In the real setup: from the trial video and a ruler, mark the sash's position whenever a
  gripper closes on or lets go of the grip bar, whenever another part of the robot starts or stops touching the sash, its
  glass or the frame, and whenever the sash starts closing or stops; add the openings made while counting and subtract
  every closing, measuring from the left jamb to the sash's left edge.
- **Torn:** the handle has given 18 mm along the sash, which takes about 14 N, at any point in the trial. It stays torn.
- With the handle on: **1.0** at 15 cm or more pulled open (`success()` also needs the sash moving under 1 cm/s for 1 s,
  which ends the trial), **0.75** from 12 cm, **0.5** from 8 cm, **0.25** from 4 cm, **0** under 4 cm.
- **0** if the handle is torn, however far the sash is open.

## Real setup

The frame and sash are white uPVC (PVC-U) profile with 3 mm float glass, and the handle a chrome-plated zinc D-handle.
The sash's stiffness comes from a friction pad under the sill: a nylon-tipped M6 set screw with a compression spring, in
a block screwed to the frame, pressing up on the sash's bottom rail. Turn the screw until a spring scale hooked on the
handle and pulled along the track reads 10 N as the closed sash starts to move and about 7 N while it slides at about
2 cm/s. Check it before each session. The handle is held on by a breakaway fixing (a magnet or a calibrated shear pin)
that lets go at about 14 N.

## Pulling on past the stop

The same demo with only its aim changed (`OPEN=0.45`: 45 cm instead of 23, so the hand carries on 21 cm past where the
sash stops): the sash slid open with the handle carrying a steady 7 N, reached its 24 cm stop with the handle on, then
the hand kept pulling, the handle's give passed 18 mm (about 14 N) and it tore off. Score **0** (`task_score: 0`), even
though the sash was fully open.

## Worked example: short pulls, closing, then a push

The sequence the scoring must not reward: pull 5 cm by the handle, close the sash, four times over (20 cm pulled in
all), then push it 20 cm open with the other gripper.

| Step | Sash open | Running total |
|---|---|---|
| Pull 5 cm by the handle | 5 cm | +5 = 5 cm |
| Close it | 0 | -5 = 0 |
| (three more pull-and-close rounds) | 0 | 0 |
| Push it 20 cm open, nothing on the handle | 20 cm | +0 = 0 |

Real trials, from the video and a ruler: the pulls are counting stretches (a gripper closed on the grip bar, no other
robot part on the sash, glass or frame), each closing is subtracted, and the push is not a counting stretch, so the
total is 0 and the score is 0. Simulation: `test_task.py`
(`test_short_pulls_closed_in_between_then_a_push_cannot_earn_credit`) plays the same sequence through `score()` and
gets 0; `test_one_gripper_lightly_tensioning_the_handle_while_the_other_pushes_earns_nothing` and
`test_the_other_gripper_touching_the_sash_glass_or_frame_earns_nothing` check the other bypasses.

## Runtime and seeds

- Run limit 1,200 steps (1 minute at 20 Hz).
- Wall time per trial on one RTX GPU, including Isaac Sim start-up: about 2 minutes for the scripted demo and 6 for a
  Dream-1 trial.
- The layout is fixed with `-E object_jitter=0`, so every trial starts the same. There are no seeds.
- Set `WINDOW_TRACE=/some/file.jsonl` to log the sash's opening, the credited opening and the handle's give every 10
  scorer calls (for debugging; it doesn't change the trial).

## Offline checks (no GPU)

```bash
git clone https://github.com/kairosiann/yam-isaac-lab-arena ~/yam-isaac-lab-arena
uv run --python 3.12 --with numpy --with matplotlib --with pytest python -m pytest test_task.py
uv run --python 3.12 --with numpy --with matplotlib python preview.py ~/slide-stiff-window-preview.png
```
