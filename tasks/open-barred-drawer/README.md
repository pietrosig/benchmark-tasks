# Open a drawer held shut by a bar

A desktop drawer cabinet stands between the arms, its three drawers facing the robot. A red bar runs across the
drawer fronts through two closed sleeves on the cabinet's front corners, 2 cm in front of the top drawer. The
instruction is only "Open the top drawer." Pulling the handle moves the drawer 2 cm until it meets the bar, and
pulling harder does not help: the robot has to notice the bar, slide it out of both sleeves along its length, set it
aside, and then pull the drawer out at least 18 cm. A robot that does nothing scores 0, and so does one that only
pulls.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()`, `score()`, `distance()` and `demo()` |
| `drawerkit.py` | Fixed furniture with sliding drawers (prismatic joints) for yam_arena task files |
| `test_task.py` | Offline checks: the file loads, the scoring rules, the geometry, the demo's reach and joint limits |
| `preview.py` | Offline picture of the layout and the demo (top, side and front views), no GPU needed |

`drawerkit.py` hooks `yam_arena.scene` when it is imported, without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment and the `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Open the top drawer." --scorer task_score \
  --max-steps 1200 --store-frames --log-dir ~/yam-eval/barred-drawer-dream
```

`task_score` is 1.0 when the trial ends in success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Open the top drawer." --scorer task_score \
  --max-steps 1200 --store-frames --log-dir ~/yam-eval/barred-drawer-demo
```

Open loop. The left arm grips the bar beside the left sleeve and slides it 16 cm to the left, out of the right
sleeve, lets go, grips it again beside the left sleeve, slides it on until it clears the left sleeve (40 cm in all),
and lays it on the table. Meanwhile the right arm waits in front of the top handle; once the bar is out it grips the handle's ends and
pulls the drawer out 19 cm. The demo succeeds after 524 steps (26 s); `task_score: 1`.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **1.0** when the top drawer is out at least 18 cm. `success()` also needs it moving under 2 cm/s, held for 0.5 s,
  which ends the trial.
- Otherwise **0.5** when the bar is out of both sleeves: its bounding box overlaps neither sleeve's slot.
- Otherwise **0**.

## Runtime and seeds

- Run limit 1,200 steps (1 minute at 20 Hz).
- Wall time per trial on one RTX GPU, including Isaac Sim start-up: about 3 minutes for the scripted demo (it ends
  at success) and about 7 minutes for a Dream-1 trial that runs to the step limit (about 0.65 s per Dream-1 request).
- The layout is fixed with `-E object_jitter=0`: the cabinet is anchored and only the bar is loose, so every trial
  starts the same. There are no seeds.

## Offline checks (no GPU)

```bash
git clone https://github.com/kairosiann/yam-isaac-lab-arena ~/yam-isaac-lab-arena
uv run --python 3.12 --with numpy --with matplotlib --with pytest python -m pytest test_task.py
uv run --python 3.12 --with numpy --with matplotlib python preview.py ~/barred-drawer-preview.png
```
