# Disconnect a full compressed-air tank without letting it shoot away

A small compressed-air reservoir (0.5 litre, aluminium, about 450 g, charged to about 6 bar) lies on the table, its rigid brass filler pipe
plugged into a push-button quick coupler on a wall air outlet. The coupler is a standard non-venting one: press its button
while nobody holds the tank and the air rushing out of the open pipe drives the tank backwards across the table: it shoots
away. The button is 19 cm from the tank's handle, so one hand can't do both. The robot has to hold the tank by its handle
with one hand, press the coupler's button with the other, keep holding while the air runs out, and then carry the tank to
the platform. A tank that shoots away scores 0, whatever happens after.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the rig, `success()`, `score()`, `distance()` and `demo()` |
| `tankkit.py` | A fixture with moving parts (prismatic or revolute joints with springs, damping, locks, followers and a speed rule) for yam_arena task files |
| `test_task.py` | Offline checks: the file loads, the scoring rules, the jet, the locks, the layout, the demo's reach, smoothness and timing |

`tankkit.py` hooks `yam_arena.scene` when it is imported, without modifying yam_arena, so the task runs with the standard
`yam_arena` embodiment and the `task_score` scorer. The wall, outlet and release are one articulation fixed to the table;
the tank hangs off it on joints that let it slide (x, y, with 100 N s/m of drag) and lift (z). Until the coupler's button
is pressed 4 mm, every one of those joints is locked: the coupler holds the pipe. Once released:

- **The air** empties with a 1.5 s time constant (a hidden part rising from 0 to 1).
- **The jet** is one more joint along x, between the wall and the tank: a very soft spring aimed 10 m away from the wall
  pushes it back with a near-constant 3 N, and its aim comes back to 0 as the air runs out. Held by its handle, the tank
  barely moves. Unheld, the jet drives it back up to 18 cm, faster than the 30 cm/s that counts as shooting away.

The kit also puts every part at its starting position right after each reset, so the first frame shows the start state.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The air tank is full. Disconnect it from the wall outlet and put it on the platform." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/air-tank-dream
```

`task_score` is 1.0 when the trial ends in success, otherwise `score()` at the end of the episode.

One Dream-1 (`dreamlabs/dream-1-beta`) trial with this command scored **0**: it ran the full 1,200 steps without
disconnecting the tank. The right arm reached over to the coupler and the tank and hovered there; the coupler's button
was never pressed, so the tank stayed connected and never moved. The left arm barely moved.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The air tank is full. Disconnect it from the wall outlet and put it on the platform." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/air-tank-demo
```

Open loop, both arms. The left arm comes down onto the tank's carry handle and grips it while the right waits out to the
side. Then the right arm comes over the coupler, presses its button 4 mm and moves away, while the left keeps holding the
tank for 3 s as the air runs out. The left arm then lifts the tank, carries it at 2 cm/s onto the platform, lets go and
moves away. With this command the demo succeeds after 660 steps (33 s of video): the tank on the platform 0.2 cm from
its centre, never moving faster than 2.4 cm/s on its own (the jet pushed the held tank back 4.1 cm at most)
(`outcome: succeeded`, `task_score: 1`), well within the 1,200-step limit.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Shot away:** the jet drove the tank back faster than 30 cm/s at any point in the trial (in the real setup: more than
  1.5 cm between two frames of a 20 fps video). It stays shot.
- **0** if it shot away, whatever happens after.
- **1.0** once the tank rests upright on its feet on the platform with its centre within 2 cm of the platform's centre (`success()` also
  needs it at rest, under 1 cm/s, for 0.5 s, which ends the trial).
- **0.5** if it was disconnected (the release pressed, the air running out) but is not on the platform.
- **0** if it is still in the coupler.

## Real setup

A 0.5-litre aluminium low-pressure air reservoir (14 x 7 cm, about 450 g; small reservoirs like this are sold for pneumatics, rated to 10
bar or more), charged to about 6 bar, with a brass valve and a rigid brass filler pipe ending in a standard 1/4-inch
industrial plug with a 3 mm orifice restrictor, seated in a standard
non-venting push-button quick coupler (the release button on the coupler itself) on a steel wall outlet. Don't use a
self-venting "safety" coupler: it empties the line before it lets go, which is exactly what this task tests the robot for. Through the 3 mm restrictor at 6 bar the jet pushes with about 5 N at first, within
what the arm can hold, and fades within about half a second; an unheld tank shoots back. The carry handle is a steel bar on two posts, clamped onto the tank
with two hose clamps. Keep people out of the area behind the tank
and fit a soft stop 20 cm behind it. Film the trial at 20 fps or more to check the 30 cm/s rule.

## Tests

```bash
YAM_ARENA=~/yam-isaac-lab-arena python -m pytest test_task.py
```

19 offline checks, no GPU needed.
