# Sort kiwis by squeezing them gently: the ripe ones into the bowl, the hard ones into the paper bag

Eight green kiwis lie in two foam-board pocket trays, four on each side of the table. Four are ripe and four are still
hard, and they look the same: kiwis barely change colour as they ripen, so the only way to tell is the squeeze everyone
uses at the shop (a ripe kiwi "gives slightly to gentle pressure"). The robot has to squeeze each kiwi gently in its
pocket, feel whether it gives, and put the ripe ones in the white bowl and the hard ones in the paper bag. Which four are
ripe changes every trial (70 layouts), so nothing can be memorised. Squeezing too hard crushes a ripe kiwi: a full grip
from the YAM gripper does, and a crushed kiwi counts for nothing.

Files: `task.py` (the scene, the scoring and the scripted demo), `fruitkit.py` (soft fruit: compliant-contact
ellipsoids, a contact view reporting the force each robot link puts on each kiwi, the simulated clock, and the bowl),
`test_task.py` (offline checks, including the exact chances of guessing).

## Changes in this revision (answering the review)

| Review point | What changed in the task |
|---|---|
| Kiwis could be assigned without sensing firmness | A kiwi now counts only if it was **squeezed in its pocket first**: both pads of one gripper closed on it (in simulation, each pad pressing at least 1 N; on a real rig, the finger gap at or below the kiwi's caliper size across the grasp) for at least 0.5 s before it left its pocket. A kiwi placed without that squeeze earns nothing, though one in the wrong container still counts against. Squeezing at full force doesn't work either: it crushes the ripe kiwis, which then score nothing. `test_task.py` checks each of these. |
| Random assignment averaged 0.5 | **Eight kiwis (four ripe) instead of four**, 70 layouts instead of 6, and credit starts only once right placements outnumber wrong ones by more than four: score = (right − wrong − 4) / 4, clipped to 0..1. Guessing now averages at most 0.036 (exact numbers below, all checked in `test_task.py`). |
| README claimed guessing averages zero | Corrected: the exact chances are in the table below. |
| Firmness of each trial kiwi not established | Every kiwi gets a quick (about 20 s), **non-destructive give test** before each session: the force needed to press it 1.5 mm, read on a kitchen scale under a screw press made from standard parts. Ripe: 150–450 g; hard: 2,000 g or more. Tested at 20 ± 3 °C, at least 2 h out of the fridge. Storage, how long kiwis stay in range, and when to replace them are below. |
| Pad covers were optional | **Nitrile finger cots on both grippers are now mandatory**: new ones each session, and replaced straight away after any trial in which a kiwi split or leaked. |

## Run a trial
On a machine set up with the official simulator (`yam-isaac-lab-arena`), from this folder:

```bash
INSTR="Eight kiwis lie in the two trays: four are ripe and four are still hard, and they look the same. Squeeze each one gently in its tray to feel which it is, without crushing it. Then put the ripe kiwis in the white bowl and the hard ones in the paper bag."

# Dream-1 (TASK_SEED picks which four kiwis are ripe: 0-69, see below)
TASK_SEED=28 inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "$INSTR" --scorer task_score --max-steps 3600 --store-frames --log-dir runs/dream

# The scripted demo, which squeezes every kiwi in its pocket and sorts it
TASK_SEED=28 inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "$INSTR" --scorer task_score --max-steps 3600 --store-frames --log-dir runs/demo

# The crush check: the same demo gripping every kiwi with the gripper's full force
TASK_SEED=28 HARD_GRIP=1 inspect-robots run ... (as for the demo)

# Offline checks (no simulator)
YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q test_task.py
```

The run prints `task_score`. The trial ends with success once all eight kiwis have been right for 1 s; otherwise it
scores `score()` at the last step. Set `TASK_TRACE=<file>` to log the kiwis, their squeeze time, the contact forces on
them and both grippers every third control step.

## Seeds: which kiwis are ripe
Pockets (cm from the midpoint between the arm bases, x forward, y to the robot's left), two rows of two on each side:
L1 (23, 14), L2 (23, 22), L3 (32, 14), L4 (32, 22) on the left; R1 (23, −14), R2 (23, −22), R3 (32, −14), R4 (32, −22)
on the right (1 near inner, 2 near outer, 3 far inner, 4 far outer). `TASK_SEED` k picks the k-th way of choosing the
four ripe pockets, in this order (the listed pockets hold the ripe kiwis; the other four the hard ones):

```
0: L1 L2 L3 L4    | 14: L1 L2 R3 R4   | 28: L1 L4 R2 R3   | 42: L2 L3 R2 R3   | 56: L3 L4 R1 R3
1: L1 L2 L3 R1    | 15: L1 L3 L4 R1   | 29: L1 L4 R2 R4   | 43: L2 L3 R2 R4   | 57: L3 L4 R1 R4
2: L1 L2 L3 R2    | 16: L1 L3 L4 R2   | 30: L1 L4 R3 R4   | 44: L2 L3 R3 R4   | 58: L3 L4 R2 R3
3: L1 L2 L3 R3    | 17: L1 L3 L4 R3   | 31: L1 R1 R2 R3   | 45: L2 L4 R1 R2   | 59: L3 L4 R2 R4
4: L1 L2 L3 R4    | 18: L1 L3 L4 R4   | 32: L1 R1 R2 R4   | 46: L2 L4 R1 R3   | 60: L3 L4 R3 R4
5: L1 L2 L4 R1    | 19: L1 L3 R1 R2   | 33: L1 R1 R3 R4   | 47: L2 L4 R1 R4   | 61: L3 R1 R2 R3
6: L1 L2 L4 R2    | 20: L1 L3 R1 R3   | 34: L1 R2 R3 R4   | 48: L2 L4 R2 R3   | 62: L3 R1 R2 R4
7: L1 L2 L4 R3    | 21: L1 L3 R1 R4   | 35: L2 L3 L4 R1   | 49: L2 L4 R2 R4   | 63: L3 R1 R3 R4
8: L1 L2 L4 R4    | 22: L1 L3 R2 R3   | 36: L2 L3 L4 R2   | 50: L2 L4 R3 R4   | 64: L3 R2 R3 R4
9: L1 L2 R1 R2    | 23: L1 L3 R2 R4   | 37: L2 L3 L4 R3   | 51: L2 R1 R2 R3   | 65: L4 R1 R2 R3
10: L1 L2 R1 R3   | 24: L1 L3 R3 R4   | 38: L2 L3 L4 R4   | 52: L2 R1 R2 R4   | 66: L4 R1 R2 R4
11: L1 L2 R1 R4   | 25: L1 L4 R1 R2   | 39: L2 L3 R1 R2   | 53: L2 R1 R3 R4   | 67: L4 R1 R3 R4
12: L1 L2 R2 R3   | 26: L1 L4 R1 R3   | 40: L2 L3 R1 R3   | 54: L2 R2 R3 R4   | 68: L4 R2 R3 R4
13: L1 L2 R2 R4   | 27: L1 L4 R1 R4   | 41: L2 L3 R1 R4   | 55: L3 L4 R1 R2   | 69: R1 R2 R3 R4
```

For real trials draw a new number 0–69 before each trial with any random number generator (for example
`python3 -c "import random; print(random.randrange(70))"`). The submitted videos use TASK_SEED 28 (ripe: L1, L4, R2, R3).

## Scoring
At the end of the trial each kiwi is one of:

| Each kiwi | Counts as | Check on a real table |
|---|---|---|
| at rest in **its own** container, let go, **squeezed in its pocket first**, and (if ripe) never crushed | right | end photo: ripe in the bowl (centre within 9.5 cm of its axis, below the rim); hard inside the bag (below its rolled top); no gripper on it. Squeeze: the trial video shows both pads of one gripper closed on it for at least 0.5 s while it still sat in its pocket, and the gripper log shows the finger gap at or below the kiwi's caliper size across the grasp then (its width, or its length if gripped end to end) |
| at rest in **the other** container, let go (squeezed or not) | wrong | end photo |
| anywhere else (still in its tray, on the table or floor, held, rolling), in its own container but never squeezed in its pocket, or a crushed ripe kiwi in the bowl | nothing | end photo / video |

**Score = (right − wrong − 4) / 4, clipped to 0..1**: 0.25 when the right placements outnumber the wrong ones by 5,
0.5 by 6, 0.75 by 7, and 1.0 with all eight right. So the robot earns credit only by sorting well beyond chance; one
misjudged kiwi among eight halves the score (7 − 1 = 6: 0.5), two cost all of it. Doing nothing, or all eight in one
container, scores 0.

A ripe kiwi is **crushed** if the gripper squeezes it more than 6 mm below its caliper size across the grasp (its width,
or its length if gripped end to end: the gripper's logged finger gap against the size on the trial sheet), or its skin
splits or juice shows; in the simulation, if any robot link presses it
with more than 15 N (a 6 mm squeeze on the model kiwi). A crushed ripe kiwi counts as nothing in the bowl and as wrong in
the bag. Hard kiwis can't be crushed by the gripper.

In the simulation a kiwi is "squeezed in its pocket" once both finger pads of one gripper have pressed it with at least
1 N each for 0.5 s in all while its centre was within 2.5 cm (horizontally) of where it started and less than 1 cm
above it; "let go" when no robot link touches it (PhysX contact view) and "at rest" under 2 cm/s.

### Guessing: the exact chances
A robot that squeezes every kiwi gently but can't tell them apart can only guess. Over the 70 equally likely layouts
(exact fractions, computed by `test_task.py` through `score()` itself):

| Strategy | Score distribution | Mean score |
|---|---|---|
| Do nothing, or squeeze and leave them | 0 always | 0 |
| Sort without squeezing in the pockets first (any assignment) | 0 always: unsqueezed kiwis can't be right | 0 |
| Squeeze everything at full force (feels the difference, but crushes the ripe ones) | 0 at best: the four hard ones right, net 4 | 0 |
| Guess four and four (using the fact that four are ripe) | 1.0 with chance 1/70 (1.4%), else 0 | 0.014 |
| Guess five in one container, three in the other (the best blind split) | 0.5 with chance 1/14 (7.1%), else 0 | 0.036 |
| Toss a coin for each kiwi | 1.0 with chance 1/256 (0.4%), 0.5 with chance 8/256 (3.1%), else 0 | 0.020 |
| All eight in one container | 0 always | 0 |

No blind strategy averages more than 1/28 (0.036), reaches full score more than 1 time in 70, or scores anything more
than 1 time in 14. (The previous version of this task claimed guessing averaged 0; with four kiwis and its scoring it
didn't.)

## Simulation
- Kiwis: rigid ellipsoids, 68 x 52 x 52 mm, 100 g, convex-hull colliders, friction 0.9. Their contacts are PhysX
  compliant contacts: ripe 5,000 N/m per contact (damping 30 N s/m), hard 100,000 N/m (150 N s/m). Closing the YAM
  gripper with each pad driven 2 mm past first touch (the gripper's 5,000 N/m finger drive in series with the contact)
  squeezes a ripe kiwi with about 6 N and it gives about 1.2 mm more than a hard one, which takes about 9.5 N. That is
  enough to lift either (friction 0.9 on two pads). A full grip (the gripper's 50 N) squeezes a ripe kiwi past 15 N and
  crushes it; a hard kiwi takes it. On the give test below the model kiwis would read about 380 g (ripe) and 7.6 kg
  (hard): inside the two bands.
- Trays: fixed boxes, a 3 mm floor, outer walls 5 mm thick and 10 mm tall, dividers 15 mm (between rows) and 20 mm
  (between columns). Bowl: drawn as a smooth surface of revolution, collided as a flat disc and 32 slanted plates. Paper
  bag: fixed boxes, the rolled top as a thicker band.
- The kiwis' reported centres sit about 5 mm back and 9 mm to the right of the nominal pocket centres from the first
  step on; no check depends on it: the in-pocket squeeze rule compares each kiwi with its own first reading, and in the
  demo every kiwi ended at least 2.5 cm inside the bowl and bag limits.
- In the simulation the gripper closes across the kiwi's width; the crush rule there is the 15 N contact force.
- Measured in the simulation: the finger pads' colliders meet a kiwi 1.5 mm before the finger joints' nominal gap
  says they would (`PAD_INSET`), so the demo aims 2 mm past that. The finger cots of the real rig are not modelled.

## Demo and results
The scripted demo (`demo()`): each arm checks the four kiwis on its own side (far outer, far inner, near outer, near
inner). Over each kiwi it opens to 72 mm (clear of the neighbouring kiwis), goes down round it, closes gently (each pad
2 mm past first touch), holds the squeeze in the pocket for 1 s, lifts it with the same grip, carries it to the bowl
(ripe) or the bag (hard), lowers it to just above the rim and lets go. After its last kiwi each arm withdraws straight out over its own side. The right arm starts 5.5 s after the left so the
two never meet over the middle of the table. The plan reads which kiwis are ripe from the task's seed (a scripted demo
may read simulator state); it still squeezes every kiwi the same way.

| Run | Seed | Result |
|---|---|---|
| demo | 28 | 1.0: success at 43 s (about 860 of the 3,600 steps), both arms withdrawn to their sides by then; all eight squeezed in their pockets first. Peak squeeze 6.8–7.8 N on the ripe kiwis, 9.1–10.4 N on the hard ones; none crushed |
| crush check (HARD_GRIP=1) | 28 | 0.0: the four ripe kiwis crushed (peak 54–59 N) and worth nothing in the bowl; the four hard ones right in the bag (right − wrong = 4) |
| Dream-1 | 28 | 0.0 at the 3,600-step limit (180 s). Both grippers shut empty by 15 s. The right arm (on the right of the video) crept inward over its tray's near pockets with its gripper shut, hovering about 2 cm above and beside the near outer kiwi (R2) at 86 s (tool point 6 cm from its centre) and about 8 cm from the near inner one (R1) at 115 s, without touching either; the left arm (on the left of the video) drew back toward its base and never came within 15 cm of a kiwi. No kiwi was touched, squeezed or moved. |

Runtime: about 5-7 minutes per scripted simulated trial on one GPU including Isaac Sim's start, about 20 minutes for a
Dream-1 trial of 3,600 steps; a real trial takes at most 3 minutes plus about 2 minutes to reset.

## Real setup
| Item | What to use |
|---|---|
| Kiwis | Green (Hayward) kiwifruit of one size, e.g. Zespri Green, count 33: 62–72 mm long and 50–54 mm wide (calipers), 90–110 g (the simulated kiwi is 68 x 52 mm, 100 g, the middle of these ranges; the 7.5 cm pockets leave at least 3 mm round the longest). Four ripe and four hard per trial (give test below), plus spares. Never mark them: keep the ripe and the hard ones in two labelled boxes off camera. |
| Pocket trays | Two, from white foam board: a 3 mm floor 17.5 x 15.0 cm; outer walls of 5 mm board standing 1 cm above it; a 1.5 cm-wide divider between the rows (three 5 mm strips glued together) and a 2.0 cm divider between the columns (four strips). That gives four pockets 7.5 cm (x) by 6.0 cm (y), centred at the pocket positions above. Taped to the table. |
| Bowl | A white ceramic serving bowl about 23 cm across the rim, 11 cm across the foot and 8 cm deep, centred at (43, 0), held with removable putty. |
| Paper bag | A brown kraft paper bag, size #12 (about 18 x 11.5 cm base), its top rolled outward until it stands 12 cm tall, opening up, centred at (10.2, 0) with its long side across the table (inside x 4.5–15.9 cm, y −9 to 9 cm), its bottom taped down. |
| Gripper pad covers (mandatory) | Powder-free nitrile finger cots, size XL (e.g. Uline S-15365, XL; or Valutek VTCNRPF-XL). One per finger: rolled on over the finger's pad end so it covers the whole gripping face without folds, its open end held with one turn of 19 mm vinyl tape on the finger above the pad (never on the gripping face). New cots at the start of every session; if a kiwi splits or leaks in a trial, stop, wipe the fingers with a damp cloth and fit new cots before the next trial. No trial runs without them. |
| Give-test press | A digital kitchen scale, an M10 thumb screw with a swivel thrust pad, and a wooden bridge (below); a 500 g calibration weight and a Lee Spring LCM095D14M spring to check them. Digital calipers for the kiwi widths. |

### The give test (each kiwi, before every session; about 20 s per kiwi; non-destructive)
It measures the force needed to press the kiwi 1.5 mm, the same property the robot feels (the non-destructive
"compression force" measure for Hayward kiwifruit is the force to compress the fruit by 1.5 mm: Feng et al. 2010).

Parts: a digital kitchen scale, 5 kg capacity, 1 g resolution (any, e.g. Escali Primo P115C); a grub screw with thrust
point DIN 6332, M10 x 80 (1.5 mm pitch: one full turn moves it exactly 1.5 mm), fitted with a DIN 6311 swivel thrust pad
for M10 (a flat steel pad, about 25 mm across, that doesn't turn on the fruit) and a star knob (e.g. Elesa-Ganter DIN
6332 and DIN 6311); a hardwood board 30 x 6 x 2 cm with an M10 threaded insert in its middle, resting on two 12 cm-tall
wooden blocks that stand on the table either side of the scale (the bridge must not touch the scale). Mark the knob so a
full turn can be counted.

Calibration at the start of every session (2 minutes): put a 500 g (or 1 kg) calibration weight on the scale; it must
read within ±2 g (±3 g). Then check the whole rig with a catalogue compression spring of known rate, Lee Spring
LCM095D14M (music wire, 7.5 mm outside diameter, 40 mm free length, rate 2.16 N/mm): stand it upright on the scale in a
10 mm hole in a scrap of foam board, run steps 2-4 below on it, and the reading at one turn must be 330 g ± 10%
(300-365 g); if not, the bridge or the screw is loose or the scale is off.

1. Bring the kiwis to room temperature: 20 ± 3 °C, at least 2 h out of the fridge (cold fruit reads firmer).
2. Lay the kiwi on its side in the middle of the scale, under the pad, and tare the scale.
3. Turn the screw down until the scale first reads 10 g or more (the pad has met the skin).
4. Turn exactly one more full turn (1.5 mm), wait 3 s, read the scale, and turn the screw back up. For a hard kiwi stop
   as soon as the reading passes 2,000 g.

| Reading at 1.5 mm | Kiwi |
|---|---|
| 150–450 g (about 1.5–4.4 N) | ripe: use as a ripe kiwi |
| 2,000 g or more (20 N or more) | hard: use as a hard kiwi |
| under 150 g, or 450–2,000 g | not used in trials (over-ripe, or part-ripe) |

These bands are this task's own specification: they define which kiwis count as ripe and hard for it. Published work
gives the method (force at 1.5 mm, Feng et al. 2010) but not values for this rig, so the bands were set from a
contact-mechanics (Hertz) estimate for flat faces on kiwi flesh, consistent with the simulated kiwis (about 380 g and
7.6 kg), with a factor of more than 4 between them, so a tenth of a millimetre of error in the turn, or the scale's own
give, doesn't move a kiwi from one band to the other. The press loads a 25 mm pad with at most 20 N for a few seconds
and leaves no mark.

Why the ripe band stops at 450 g: for a body pressed between flat faces, Hertz contact gives F = K d^1.5. The firmest
ripe kiwi allowed (4.4 N at d = 1.5 mm) has K = 4.4 / 1.5^1.5 = 2.4 N/mm^1.5, so the YAM's full grip of about 40 N
squeezes it d = (40 / 2.4)^(2/3) = 6.5 mm: past the 6 mm crush limit, so a full-force grip can't pass as gentle on any
ripe kiwi in the band (softer ones give more; real flesh also yields, which only adds to it). The softest hard kiwi
allowed (19.6 N at 1.5 mm, K = 10.7) gives 2.4 mm at 40 N, well short of 6 mm, and at a gentle 5 N the ripe kiwis give
1.6 mm or more against 0.6 mm or less for the hard ones: the "about 1 mm of extra give" the robot has to feel.

### Storage and handling
- Buy hard kiwis and keep them at 0–4 °C in the fridge: hard Hayward kiwis stay firm for weeks in cold storage (UC Davis:
  8–12 weeks to soften to 5 lbf flesh firmness at 0 °C in air).
- To ripen some, leave them at room temperature in a closed paper bag with an apple or banana (Zespri: 24–48 h; 3–5 days
  without the fruit) and give-test them daily; move each one to the fridge as soon as it reads 150–450 g. Ripe kiwis keep
  up to 5 days in the fridge (Zespri Green).
- On a trial day: take the kiwis out 2 h before, give-test all of them at 20 ± 3 °C, and use only those in range. A
  session runs at most 4 h at room temperature; then the kiwis go back in the fridge and are tested again before the
  next session.
- During a session, give-test every kiwi again every hour, or after every 10 trials it has been used in, whichever comes
  first, and at once if it was dropped. Replace any that falls out of its band.
- Sizes for the squeeze and crush checks: as each kiwi is laid out, measure its width across the middle and its length
  end to end with digital calipers and write both on the trial sheet by pocket (L1-L4, R1-R4). The check uses the size
  across the grasp: the width when the fingers close across the kiwi (as they do lying long axis forward), the length if
  the robot grips it end to end. Once per session, with the finger cots on, close
  each gripper on a 50 mm reference (the calipers' jaws set to 50.0 mm, or a 50 mm gauge block) and check its reported
  finger gap reads 50 mm; note any offset on the sheet and apply it to that session's gap readings.
- Replace a kiwi when it falls out of its range, was dropped on the floor, shows a split, bruise or wrinkled skin, or was
  crushed in a trial.

Reset between trials (about 2 minutes): take all kiwis out of the bowl, bag and trays into their two boxes (ripe, hard),
draw a new TASK_SEED, and lay the four ripe and four hard kiwis in the pockets it names, long axis forward, writing each
one's caliper width and length on the trial sheet by pocket. Check the finger cots are intact; replace them if torn or wet.

In the screenshot and videos the right tray's outer pockets look darker: that is the right arm's shadow (lighting only);
both trays and all eight kiwis are the same.

Safety: kiwis, a paper bag and a ceramic bowl fixed to the table; nothing sharp, hot or heavy. The pads are covered.

## Sources
- Zespri, how to tell when a kiwifruit is ripe ("gently press ... if it gives to slight pressure"), ripening (24–48 h in
  a paper bag with a banana or apple; 3–5 days on the counter) and storage (ripe Green: up to 5 days in the fridge):
  https://www.zespri.com/en-US/blogdetail/how-to-store-kiwi-fruit-for-freshness-and-flavor
- Compression force (force to compress a Hayward kiwifruit by 1.5 mm) as a non-destructive firmness measure: Feng J.,
  Mackay B.R., Maguire K.M., Jeffery P.B. (2010), Acta Horticulturae 880: 283–289, https://ishs.org/ishs-article/880_33/
- Non-destructive 1 mm compression firmness of kiwifruit; eating ripeness at 1 kgf flesh firmness or less: Massey
  University, https://mro.massey.ac.nz/handle/10179/70560
- Cold-storage softening of kiwifruit by size (UC Davis):
  https://crisosto.ucdavis.edu/sites/g/files/dgvnsk6166/files/inline-files/043-99-Kiwi-Size-influences-softening-rate-during-storage.pdf
  ; maturity and quality (ripe at 2–3 lbf flesh firmness): https://postharvest.ucdavis.edu/node/5301
- Calibration spring, Lee Spring LCM095D14M (rate 2.163 N/mm):
  https://www.leespring.com/product/compression-spring-lcm095d14m-music-wire
- Nitrile finger cots: https://www.uline.com/BL_6527/Nitrile-Finger-Cots ,
  https://www.tequipment.net/Valutek/VTCNRPF-XL/Cleanroom-Fingercots/
- DIN 6332 grub screws with thrust point and DIN 6311 thrust pads:
  https://www.ganternorm.com/en/products/3.2-Mounting-positioning-levelling-with-screws-clamping-and-supporting-elements/Thrust-pads/DIN-6311-Thrust-pads-Steel

No third-party meshes or scans: every shape is built in `task.py` and `fruitkit.py`.
