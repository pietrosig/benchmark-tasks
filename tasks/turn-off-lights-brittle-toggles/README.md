# Turn off the lights on an old panel with brittle toggle switches

An old lighting panel is fixed to the table: a wooden board with four brass toggle switches in a row and a lamp in
front of each. A lever leaning away from the arms is on, towards them off. Three lamps are lit; the third toggle from the
right is already off. A lever tips with about 0.2 N and, just past upright, clicks over by itself and its lamp goes out.
The levers are brittle: forced on past their end stop with more than about 3 N they snap off at the base. The robot has
to see which lamps are lit, flip only those, each just far enough to click over, and leave the one already off alone.
Pushing every lever over hard, or shoving the one that is already off, snaps a lever and scores 0.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the panel, `success()`, `score()`, `distance()` and `demo()` |
| `togglekit.py` | A fixture with moving parts (prismatic or revolute joints with springs, damping, a break point, a click-over toggle) for yam_arena task files |
| `test_task.py` | Offline checks: the file loads, the scoring rules, the forces, the geometry, the demo's reach, joint limits and smoothness |

`togglekit.py` hooks `yam_arena.scene` when it is imported, without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. The panel is one articulation fixed to the table. Each toggle
is a revolute joint with a spring (0.02 N m/rad) that holds the lever on its side; once the lever is pulled 1.7 degrees
past upright the spring's target switches to the other side, so it clicks over. Each lever sits on a second revolute
joint at its base, a stiff spring (2 N m/rad) that stands in for the brittle brass: bent 3 degrees (about 3 N at the
knob) it snaps, going limp for good so the lever falls over. A lamp going out is a dark shade that rises over its bulb as
its toggle goes off. The kit puts every part at its starting position right after each reset, so the first frame and
the first observation already show the start state.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The panel's switches are old and brittle. Turn off the lights." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/turn-off-lights-dream
```

`task_score` is 1.0 when the trial ends in success, otherwise `score()` at the end of the episode.

One Dream-1 (`dreamlabs/dream-1-beta`) trial with this command scored **0**: it ran the full 1,200 steps without
switching any lamp off. Both arms moved about beside the ends of the panel; the leftmost lever was nudged 1.6 degrees
(it only clicks over once tipped about 22, just past upright) and sprang back, and no lever snapped (bent 0.12 degrees at most). It made 25 requests
in one Dream session, each on the first attempt, and took about 3 minutes.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The panel's switches are old and brittle. Turn off the lights." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/turn-off-lights-demo
```

Open loop, both arms, each in one continuous motion after the first move from the park pose. The right arm switches off
the two lit toggles on its side (y -12 and -4), the left arm the one on its side (y 12); the toggle that is already off
is left alone. For each one the hand comes down just behind the knob, pulls it towards the arms at 1.2 cm/s until just
past the click point, and lifts off while the toggle clicks over by itself. Between toggles it moves at 2.5 cm/s.
With this command the demo succeeds after 358 steps (17.9 s of video): all four lamps off, no lever snapped, the levers
bent 1.1 degrees at most (`outcome: succeeded`, `task_score: 1`), well within the 1,200-step limit.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Off:** a lever leaning more than 12 degrees towards the arms. In the real setup, read from a photo.
- **Snapped:** a lever bent 3 degrees at its base, which takes about 3 N at the knob, at any point in the trial. It
  stays snapped. In the real setup, the lever is broken off.
- **0** if any lever has snapped, including the one that was already off at the start.
- Otherwise a third of a point for each of the three lamps lit at the start that is now off: **0, 0.33, 0.67, 1.0**.
- `success()` also needs every toggle still (under 0.05 rad/s) for 1 s, which ends the trial.
- How the levers are moved isn't scored.

## Tests

```bash
YAM_ARENA=~/yam-isaac-lab-arena python -m pytest test_task.py
```

24 offline checks, no GPU needed.
