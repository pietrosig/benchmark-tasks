# Walk a drifted laser beam back through two alignment apertures with two kinematic mirrors

The ability tested is a **coupled four-adjuster walk with fractional-turn control**: two steering mirrors in kinematic
mounts, four 1/4-80 adjuster knobs, two alignment apertures (0.8 mm holes) that define the beam's axis, and a photodiode
read on a multimeter. Every knob moves the beam on both apertures, so the hands alternate between the mirrors, turning
knobs by fractions of a turn, without disturbing the post-mounted mounts, until the reading is as high as it goes. The
knobs carry no mark of the aligned setting, so the only guide is the beam and the meter. This is how free-space laser
paths are realigned in optics, physics and biophotonics labs (routinely after any change to the path in two-photon,
confocal and optical-trap setups), and it differs from trimming one adjuster against a readout.

Instruction (`instruction.txt`): *The laser beam has drifted off its two alignment apertures. Walk it back through both apertures with the two mirrors' adjuster knobs until the meter reads as high as it will go, then take your hands away from the mounts.*

## Run

```
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py --instruction "$(cat instruction.txt)" \
  --scorer task_score -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 --max-steps 3000
```

- `--policy dream` for the model run. Run limit 3,000 steps (2.5 minutes at 20 Hz); the scripted demo needs about 730
  (36 s, ending with a 6 s hold with the hands clear). Wall time on a shared GPU: about 20 minutes for the demo, about an
  hour for a full 3,000-step run.
- Seeds: `TASK_SEED=0..19` picks the starting misalignment (default 0) from the table below.
- Offline tests (no simulator): `python -m pytest -q test_task.py` (beam model, clipping at the first aperture and the
  far spot and meter going dark with it, every seed starts at 0, the partial-credit levels, the knob ticks not marking the
  aligned setting, keep-out and hands-away rules, the demo's waypoints all reachable by IK).
- Debug: `TASK_TRACE=<file>` logs the reading, aperture 1's transmission, whether the far spot is lit, both offsets, the
  knob angles, the tool points and the grippers' opening every 10 calls. `PROOF=<knob>` (e.g. `PROOF=m1_side`) makes the
  demo turn one knob 0.47 turn on its own.

## Scene (positions from the midpoint between the arm bases: x forward, y to the left arm, heights above the bench top)

Beam height 200 mm. Everything stands on a 400 x 460 mm piece of 1/2 in (12.7 mm) plywood, painted matte black, centred
at (0.40, 0.00) and clamped to the bench with two C-clamps.

| Item | Position | Product |
|---|---|---|
| Laser module, 650 nm, 1 mW, Class 2, collimated dot, Ø9 x 20 mm, fires along -x | exit at (0.56, +0.11) | Picotronic LFD650-1-12(9x20)-C400-JST (10.8-13.2 V), in a printed holder on a printed stand (`stands.scad`, part `laser`), 12 V DC from a Mean Well GST18U12-P1J plug-in adapter through an Adafruit 368 jack-to-screw-terminal adapter |
| Mirror M1, normal at -45 deg (faces +x, -y) | (0.28, +0.11) | Thorlabs KM100 + ME1-G01 Ø1" protected-aluminium mirror, on a TR6 post in a PH3 post holder |
| Mirror M2, normal at +45 deg (faces +x, +y) | (0.28, -0.11) | same |
| Aperture 1: white plate 40 x 40 x 3 mm, 0.8 mm hole at beam height | (0.40, -0.11) | printed (`stands.scad`, part `aperture`) |
| Aperture 2: the same | (0.52, -0.11) | same |
| Photodiode in a black holder, behind aperture 2, with a black light-shroud tube (Ø16 mm outside, Ø10 mm inside, 31 mm long) reaching to 1 mm behind aperture 2's plate | (0.555, -0.11) | Vishay BPW34 in a printed holder with the shroud (`stands.scad`, part `photodiode`) |
| Multimeter, face up, 2000 µA DC range, reading the photodiode's short-circuit current | (0.45, -0.25) | UNI-T UT33D+ (about 130 x 74 x 35 mm) |

Each KM100's two adjuster knobs (Ø12 mm, 1/4-80 thread: 0.318 mm per turn, about 8 mrad of mirror tilt per turn on
the mount's lever, an estimate used in the simulation) face the robot from the back of the mount, at the plate's corners
12.7 mm up-and-left and 12.7 mm down-and-right of the mirror's centre (seen from behind): the upper knob tips the mirror up
and down, the lower one turns it left and right. **No marks:** the knobs and mounts are as supplied (black, unmarked);
nothing on the bench shows the aligned setting. Each PH3 is fixed by a 1/4"-20 x 1-1/4" cap screw from under the board
into its tapped base; set the TR6 in it so the mirror centre is at beam height. The printed stands are screwed down with
two 4 mm wood screws each.

**Detector.** The BPW34 is wired straight to the multimeter's µA input (short-circuit photocurrent, linear in the
optical power; through the two 0.8 mm holes the aligned reading is some tens of µA, on the 2000 µA range with 1 µA
resolution). The dark reading (laser blocked) is subtracted from every reading.

**Before each trial (reset, about 5 minutes; nothing is consumed).** With the arms in their ready pose, block the laser
and note the dark reading. Unblock and realign by hand: walk the beam with the four knobs (alternating the mirrors) to the
highest reading, and note and log it as P_ref. If P_ref is below 90% of the bench's first logged P_ref, clean the
mirrors and realign again. Then detune each knob by the seeded number of turns from the table (signed: + is counter-clockwise seen from behind
the mount): stick a small masking-tape flag on the knob's rim as a pointer, hold a printed 10-degree dial card behind the
knob (a Ø40 mm paper disc with 36 radial ticks, slit to its centre so it slides round the adjuster), count whole turns by
the flag, set the fraction against the card, then remove the flag and the card. The beam then lands on aperture 1's
plate beside the hole and the meter reads 0 (if it doesn't, use the next seed). Because the aligned setting is found
afresh by the meter at every reset and nothing marks it, the robot cannot restore known knob positions: it has to walk
the beam.

Seeded starting offsets (turns from the aligned setting; `task.OFFSETS`):

| Seed | M1 upper | M1 lower | M2 upper | M2 lower |
|---|---|---|---|---|
| 0 | -0.60 | -0.46 | -0.33 | +0.48 |
| 1 | +0.37 | +0.56 | +0.37 | +0.51 |
| 2 | -0.31 | +0.33 | +0.53 | -0.58 |
| 3 | -0.43 | -0.31 | -0.50 | -0.49 |
| 4 | -0.47 | -0.56 | -0.37 | -0.34 |
| 5 | +0.57 | -0.46 | -0.32 | +0.47 |
| 6 | +0.31 | +0.59 | -0.48 | -0.39 |
| 7 | +0.47 | +0.34 | +0.35 | +0.39 |
| 8 | +0.42 | +0.39 | +0.41 | +0.55 |
| 9 | -0.47 | +0.46 | +0.42 | +0.39 |
| 10 | -0.54 | -0.40 | -0.35 | +0.44 |
| 11 | +0.48 | +0.56 | +0.51 | +0.57 |
| 12 | +0.59 | +0.46 | -0.57 | -0.49 |
| 13 | -0.45 | +0.56 | +0.48 | -0.56 |
| 14 | -0.56 | -0.53 | +0.47 | +0.49 |
| 15 | -0.56 | +0.45 | -0.46 | +0.31 |
| 16 | -0.38 | +0.55 | +0.37 | -0.40 |
| 17 | -0.33 | +0.57 | +0.35 | +0.48 |
| 18 | +0.44 | -0.36 | +0.51 | -0.37 |
| 19 | +0.35 | +0.57 | +0.41 | +0.53 |

## Scoring (one rule, simulation and real trial)

Measured at the end of the trial with the robot's hands away:

- **0** if at any time any gripper or robot link touched the laser, either aperture plate, the photodiode or a mirror's
  face, or visibly moved or rotated a post, stand or mount body (more than 1 mm or 2 degrees against the first frame of
  the trial video; in simulation, a tool point inside a keep-out box around each), or if at the end a gripper is still within 5 cm of a knob.
- Otherwise by the photocurrent P (dark subtracted) as a fraction of P_ref: **1.0** at P >= 0.80 P_ref, **0.5** at
  P >= 0.40 P_ref, **0.25** at P >= 0.10 P_ref, **0** below.

Doing nothing scores 0 (every seed starts at 0).

## How the simulation works (fidelity)

The beam is not rendered as light. Every step, `beam()` ray-traces it from the laser off both mirror planes (exact
reflection) to the two aperture planes, from the four knob angles: one knob turn tilts its mirror 8 mrad about the
corresponding axis (estimate: the 1/4-80 thread's 0.318 mm per turn on a lever of about 40 mm). Each aperture's
transmission is exp(-2 r² / w²) for the beam centre's offset r from its hole, with the module's 1/e² radius w = 1.5 mm.

**Clipping at aperture 1.** The beam is physically stopped by aperture 1's plate: when its centre is more than one beam
radius outside the hole (r1 > 0.4 + 1.5 = 1.9 mm), aperture 1 passes nothing (T1 = 0), no spot is drawn on aperture 2
(the far spot is pushed back inside its plate) and the meter reads 0. Otherwise the reading is T1 x T2, relative to the
aligned reference, and the far spot is drawn where the beam lands. If the beam missed aperture 1's plate altogether it
would travel on to aperture 2 (drawn there), but the photodiode, behind both holes, still reads 0. Every seeded start has
the beam on aperture 1's plate beside the hole (r1 = 2.2-4.4 mm), so only the near spot is lit (test
`test_far_spot_and_meter_dark_when_near_aperture_blocks`).

Remaining approximations: no diffraction at the holes, and aperture 2's transmission uses the unclipped beam's radius;
the real beamlet behind a 0.8 mm hole is narrower (about 0.5 mm), so on the real bench the far aperture's tolerance is
tighter than simulated (the score is relative to P_ref, measured on the real bench). The spots are drawn 6 mm across so
they can be seen and are not dimmed with the transmission; the meter's reading is a green bar. The beam from the laser to
M1 to M2 is drawn as a red line. The knobs are real revolute joints (0.03 N·m friction) that the grippers grip and turn;
the mounts, posts, stands and the photodiode's shroud are fixed. The touch rules use tool-point keep-out boxes (laser, aperture plates, photodiode,
mirror faces, each mount's body and post). The mirror plates do not visibly tilt (a few mrad). Each knob carries a small
grey tick, drawn only so its turning shows in the video: it sits at a random angle per seed and knob, unrelated to the
aligned setting (the real knobs are unmarked).

## Demo

`demo()` (policy `yam_scripted`) reads the starting knob angles from the simulator (allowed for scripted demos) and
turns each back to zero: the left arm does M1's two knobs, the right arm M2's. Each knob is gripped from behind along
its axis and turned by rolling the wrist (up to about 230 degrees in one grasp; longer turns take a second grasp);
between knobs the wrist turns back and the hand returns to its start point. At the end both hands move clear and hold
for 6 s while the meter is read. Result: seed 0, `task_score` 1.0: reading 0.97 of the aligned reference, every knob
within 0.02 turn of aligned. The far spot first lights at about 27 s, when M1's knobs bring the beam into aperture 1's
hole (r1 = 1.9 mm); success (reading >= 0.80, hands clear) holds from about 32 s, and the run ends at 36 s (726 control
steps, a quarter of the run limit).

## Dream-1

Dream-1 (seed 0, same instruction, 3,000-step limit): `task_score` 0.0. The right arm moved toward M2's knobs
but never gripped or turned one; the left arm stayed near its start; no knob moved and the meter read 0 throughout, with
only the near spot lit (`out_rev2/dream1.trace`).

## Safety

The laser is Class 2 (1 mW, visible): the blink reflex protects the eye. The beam runs horizontal at 200 mm above
the bench, below a standing person's eye level, and ends on the photodiode; no reflective tools are used near it.

## Real parts (about USD 290)

| Part | Qty | Unit price (USD) | Total | Source |
|---|---|---|---|---|
| Thorlabs KM100 kinematic mirror mount, Ø1", two 1/4-80 adjusters | 2 | 44.78 | 89.56 | thorlabs.com/item/KM100 |
| Thorlabs ME1-G01 Ø1" protected-aluminium mirror, 3.2 mm thick | 2 | 36.67 | 73.34 | thorlabs.com/item/ME1-G01 |
| Thorlabs TR6 Ø1/2" post, 6" | 2 | 8.66 | 17.32 | thorlabs.com/item/TR6 |
| Thorlabs PH3 Ø1/2" post holder, 3" | 2 | 10.29 | 20.58 | thorlabs.com/item/PH3 |
| Picotronic LFD650-1-12(9x20)-C400-JST dot laser, 650 nm, 1 mW, Class 2 (order no. 70124248) | 1 | about 22 (EUR 18.90 incl. VAT) | 22.00 | shop.picotronic.de/en/p/70124248 |
| Mean Well GST18U12-P1J 12 V 1.5 A plug-in adapter, 2.1 mm plug (for the laser) | 1 | 13.50 | 13.50 | masterelectronics.com |
| Adafruit 368 2.1 mm DC jack to screw-terminal adapter (DigiKey 1528-1386-ND) | 1 | 2.00 | 2.00 | adafruit.com |
| Vishay BPW34 photodiode (DigiKey 751-1015-ND) | 1 | about 1.23 | 1.23 | digikey.com |
| UNI-T UT33D+ multimeter (2000 µA DC range) | 1 | 12.39 | 12.39 | tme.com |
| 1/2 in x 2 ft x 2 ft plywood project panel (Columbia Forest Products PureBond Poplar, model 2915), cut to 400 x 460 mm | 1 | 12.34 | 12.34 | homedepot.com |
| Hardware: 2 x 1/4"-20 x 1-1/4" cap screws, 8 x 4 mm wood screws, matte black spray paint, masking tape | 1 | about 11 | 11.00 | any hardware store |
| Two 4 in C-clamps | 2 | about 5 | 10.00 | any hardware store |
| PLA filament (about 160 g, white and black) and a 0.8 mm drill bit for the holes | 1 | about 5 | 5.00 | any supplier |

Total about USD 290 (about USD 264 of it catalogue prices; the rest ordinary hardware). The printed parts (two apertures, laser holder,
photodiode holder with its light shroud) are in `stands.scad` (OpenSCAD, parametric; dimensions in the file). Optional: Thorlabs ID12 iris
diaphragms (USD 62.39 each, with a TR3 post) closed to their 0.8 mm minimum can replace the printed apertures; they
behave the same.
