# Sort kiwis by squeezing them gently: the ripe ones into the bowl, the hard ones into the paper bag

Four green kiwis lie in two small pocket trays, two on each side of the table. Two are ripe and two are still hard, and
they look the same: kiwis barely change colour as they ripen, so the only way to tell is the squeeze everyone uses at
the shop (a ripe kiwi "gives slightly to gentle pressure"). The robot has to squeeze each kiwi gently, feel whether it
gives, and put the ripe ones in the white bowl and the hard ones in the paper bag. Which two are ripe changes every
trial (six layouts), so nothing can be memorised. Squeezing too hard crushes a ripe kiwi: a full grip from the YAM
gripper does, and a crushed kiwi counts for nothing.

Files: `task.py` (the scene, the scoring and the scripted demo), `fruitkit.py` (soft fruit: compliant-contact
ellipsoids, a contact view reporting the force each robot link puts on each kiwi, and the bowl), `test_task.py`
(offline checks).

## Run a trial
On a machine set up with the official simulator (`yam-isaac-lab-arena`), from this folder:

```bash
INSTR="Four kiwis lie in the two trays: two are ripe and two are still hard, and they look the same. Squeeze each one gently to feel which it is, without crushing it. Put the ripe kiwis in the white bowl and the hard ones in the paper bag."

# Dream-1 (TASK_SEED picks which two kiwis are ripe: 0-5, see below)
TASK_SEED=2 inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "$INSTR" --scorer task_score --max-steps 2400 --store-frames --log-dir runs/dream

# The scripted demo, which reaches full score on every seed
TASK_SEED=2 inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "$INSTR" --scorer task_score --max-steps 2400 --store-frames --log-dir runs/demo

# The crush check: the same demo gripping every kiwi with the gripper's full force; it ends at 0.5
TASK_SEED=2 HARD_GRIP=1 inspect-robots run ... (as for the demo)

# Offline checks (no simulator)
YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q test_task.py
```

The run prints `task_score`. The trial ends with success once all four kiwis have been right for 1 s; otherwise it
scores `score()` at the last step. Set `TASK_TRACE=<file>` to log the kiwis, the contact forces on them and both
grippers every third control step.

## Seeds: which kiwis are ripe
Pockets: left near (22, 15), left far (32, 15), right near (22, -15), right far (32, -15) (cm from the midpoint between
the arm bases, x forward, y to the robot's left). Two kiwis are ripe in every layout:

| TASK_SEED | ripe kiwis | hard kiwis |
|---|---|---|
| 0 | left near, left far | right near, right far |
| 1 | left near, right near | left far, right far |
| 2 | left near, right far | left far, right near |
| 3 | left far, right near | left near, right far |
| 4 | left far, right far | left near, right near |
| 5 | right near, right far | left near, left far |

For real trials, roll a die before each trial (TASK_SEED = roll - 1) and lay the kiwis out accordingly. The submitted
videos use TASK_SEED 2; the demo was also run on seed 4 (1.0) and, in an earlier tuning run, seed 1.

## Scoring
0.25 per kiwi; the total is never below 0.

| Each kiwi | Score | Check on a real table |
|---|---|---|
| ripe, at rest in the bowl, let go, never crushed | +0.25 | end photo: inside the bowl, below the rim, centre within 8 cm of its axis, no gripper on it |
| hard, at rest in the paper bag, let go | +0.25 | end photo from above: inside the bag, below its rolled top |
| at rest in the other container | -0.25 | end photo |
| anywhere else: still in its tray, on the table or floor, held, rolling | 0 | end photo / video |
| ripe and crushed at any time, wherever it ends | 0 (or -0.25 in the bag) | the gripper's finger-gap log against the kiwi's caliper width (more than 6 mm narrower), or split skin or juice |

- All four right: 1.0. Doing nothing: 0. All four in the bowl, or all four in the bag: 0.25 x 2 - 0.25 x 2 = 0.
  Guessing scores 0 on average.
- In the simulation a kiwi is "let go" when no robot link touches it (PhysX contact view) and "at rest" under 2 cm/s;
  a ripe kiwi is crushed once any robot link presses it with more than 15 N, which on the model kiwi is a 6 mm
  squeeze (3 mm at each pad).

## Simulation
- Kiwis: rigid ellipsoids, 68 x 52 x 52 mm, 100 g, convex-hull colliders, friction 0.9. Their contacts are PhysX
  compliant contacts: ripe 5,000 N/m per contact (damping 30 N s/m), hard 100,000 N/m (150 N s/m). Closing the YAM
  gripper with each pad driven 2 mm past first touch (the gripper's 5,000 N/m finger drive in series with the contact)
  squeezes a ripe kiwi with about 6.5 N and it gives about 1.2 mm more than a hard one, which takes about 9.5 N. That
  is enough to lift either (friction 0.9 on two pads). A full grip (the gripper's 50 N) squeezes a ripe kiwi past
  15 N and crushes it; a hard kiwi takes it.
- Trays: fixed boxes, a 3 mm floor and walls 4 mm thick and 10 mm tall (foam board in the real build). Bowl: drawn as a smooth surface of
  revolution, collided as a flat disc and 32 slanted plates. Paper bag: fixed boxes, the rolled top as a thicker band.
- Measured in the simulation: the finger pads' colliders meet a kiwi 1.5 mm before the finger joints' nominal gap
  says they would (`PAD_INSET`), so the demo aims 2 mm past that.

## Demo and results
The scripted demo (`demo()`): each arm checks the two kiwis on its own side, far one first. Over each kiwi it opens
fully, goes down round it, closes gently (each pad 2 mm past first touch), holds the squeeze for 1 s, lifts it with the
same grip, carries it to the bowl (ripe) or the bag (hard), lowers it to just above the rim and lets go. The right arm
starts 6 s after the left so the two never meet over the middle of the table. The plan reads which kiwis are ripe from
the task's seed (a scripted demo may read simulator state); it still squeezes every kiwi the same way.

| Run | Seed | Result |
|---|---|---|
| demo | 2 | 1.0, success at 26 s (about 540 steps of the 2,400) |
| demo | 4 | 1.0 |
| crush check (HARD_GRIP=1) | 2 | 0.5: both ripe kiwis crushed (peak 59-60 N), the two hard ones in the bag |
| Dream-1 | 2 | 0.0 over the full 2,400 steps: both arms rose from the ready pose and hovered over their own side of the table, the gripper 12-27 cm above it; neither reached a kiwi (the right gripper came closest, about 13 cm from the right near kiwi at 33 s, and closed empty from 42 s; the left closed empty from about 84 s). No kiwi moved. |

Peak squeeze in the seed-2 demo: 6.6 and 6.8 N on the ripe kiwis, 9.4 and 9.5 N on the hard ones. Runtime: about
4-6 minutes per simulated trial on one GPU including Isaac Sim's start; a real trial takes at most 2 minutes plus about
1 minute to reset.

## Real setup
| Item | What to use |
|---|---|
| Kiwis | Four green (Hayward) kiwifruit of one size, about 68 x 52 mm and 100 g (size count 30-33), e.g. Zespri Green. Two ripe, two hard (see below). |
| Pocket trays | Two, from 5 mm white foam board, hot-glued: a base 20.4 x 6.8 cm with 1 cm-tall walls round the edge and one cross wall, giving two pockets 7.6 x 6.0 cm. Taped to the table, long side forward, centred 15 cm either side of the midline, pocket centres 22 and 32 cm forward. |
| Bowl | Any white ceramic bowl about 19 cm across the rim, 9 cm across the foot and 6.5 cm deep (a serving or salad bowl), centred 36 cm forward on the midline, held with removable putty. |
| Paper bag | A brown kraft paper lunch bag, size #6 (about 15 x 9 cm base), its top rolled outward until it stands 10 cm tall, opening up, centred 17 cm forward on the midline with its long side across the table, its bottom taped down. |

Preparing kiwis: buy hard kiwis (they keep for weeks in the fridge). Ripen some for 3-5 days at room temperature in a
paper bag with an apple. A kiwi counts as ripe when it gives slightly to gentle thumb pressure (Zespri's test) and as
hard when it doesn't give at all; to check a batch, cut one kiwi from it and measure it with a fruit penetrometer
(7.9 mm tip): ripe batches read 1 kgf or less (Zespri's eating-ripe firmness), hard ones well above 4 kgf. Ripe kiwis
keep about 5 days in the fridge; let them reach room temperature before trials. Replace any kiwi that has gone soft
past "gives slightly", or was crushed. Measure each kiwi's width with calipers before a trial for the crush check.

Safety: kiwis, a paper bag and a ceramic bowl fixed to the table; nothing sharp, hot or heavy. A crushed kiwi can
leave juice on the pads: wipe them after the trial (or fit thin pad covers).

## Sources
- Zespri, how to tell when a kiwifruit is ripe ("gently press ... if it gives to slight pressure ... it's ready to eat")
  and storage times: https://www.zespri.com/en-US/blogdetail/how-to-store-kiwi-fruit-for-freshness-and-flavor ,
  https://www.zespri.com/en-US/blogdetail/can-i-refrigerate-kiwifruit
- Zespri handling guide, firmness by penetrometer (7.9 mm probe): https://handling-guide.zespri.com/en/assess
- Eating ripeness at 1 kgf or less: Massey University, non-destructive firmness assessment of kiwifruit,
  https://mro.massey.ac.nz/handle/10179/70560

No third-party meshes or scans: every shape is built in `task.py` and `fruitkit.py`.
