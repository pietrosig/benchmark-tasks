# Stop a runaway hoist from its hanging pendant

An overhead hoist lowers a steel toolbox at 1.5 cm/s onto a bottle of acid; its pendant's down button is jammed.
The robot must press the emergency stop on the pendant, which hangs free on its cable, before the toolbox lands
(10 s, untouched). The pendant swings away from a push, so it has to be held or pinched while the mushroom
(20 N to latch) is pressed.

Files: `task.py` (scene, scoring, scripted demo), `hoistkit.py` (articulated fixtures: the hoist motor, the
swinging pendant, the latching stop).

## Run a trial

    inspect-robots run --embodiment yam_arena --policy <policy> -E task=task.py \
      -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
      --instruction "The hoist won't stop lowering the toolbox onto the bottle of acid. Stop it with the emergency stop on its pendant." \
      --scorer task_score --max-steps 400

Runtime per trial: about 1.5 minutes on one L40S (400 steps). No seeds: the layout is fixed (`object_jitter=0`).
Set `HOIST_TRACE=<file>` to log the gap, the stop's travel and the pendant's swing every 10 scoring calls.

## Scoring (`score()` in task.py)

- 0 once the toolbox has touched the bottle (latched for the rest of the trial).
- 0 while the emergency stop isn't latched (pushed in 3.5 mm).
- With it latched: 1.0 if the toolbox's underside is 5 cm or more above the cap, 0.5 if less.

## Scripted demo

    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py ...   (same flags)

The right hand comes down over the pendant with its fingers 9 cm open, one finger in front of the mushroom and
one behind the housing, slowly enough not to set it swinging, then closes: the gripper squeezes the mushroom in
against the housing. Result: score 1.0. The stop latched with the toolbox 6.7 cm above the cap, and the trial ended
as succeeded 5 s later with the toolbox still up.

`ROUGH=1` runs the failure for comparison: the hand jabs the mushroom with closed fingers without holding the
pendant. The pendant swings away, the mushroom goes in 0.5 mm, and the toolbox lands on the bottle: score 0.

Dream-1 (`dreamlabs/dream-1-beta`) result with this command: score 0. It never touched the pendant (the stop moved 0 mm, the pendant swayed 0.2 degrees at most), and the toolbox came down on the bottle about 10 s in.
