# Walk a drifted laser beam back through two irises with two kinematic mirrors

The ability tested is a **coupled four-adjuster walk with fractional-turn control**: two steering mirrors in kinematic
mounts, four 1/4-80 adjuster knobs, two irises that define the beam's axis, and a photodiode read on a multimeter. Every
knob moves the beam on both irises, so the hands alternate between the mirrors, turning knobs by fractions of a turn,
without disturbing the post-mounted mounts, until the reading is as high as it goes. This is how free-space laser paths
are realigned in optics, physics and biophotonics labs (routinely after any change to the path in two-photon, confocal
and optical-trap setups), and it differs from trimming one adjuster against a readout.

Instruction (`instruction.txt`): *The laser beam has drifted off its two irises. Walk it back through both irises with
the two mirrors' adjuster knobs until the meter reads as high as it will go, then take your hands away from the mounts.*

## Run

```
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py --instruction "$(cat instruction.txt)" \
  --scorer task_score -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 --max-steps 3000
```

- `--policy dream` for the model run. Run limit 3,000 steps (2.5 minutes at 20 Hz); the scripted demo needs about 730
  (36 s, ending with a 5 s hold with the hands clear). Wall time on a shared GPU: about 20 minutes for the demo, about an
  hour for a full 3,000-step run.
- Seeds: `TASK_SEED=0..19` picks the starting misalignment (default 0) from the table below.
- Offline tests (no simulator): `python -m pytest -q test_task.py` (beam model, every seed starts at 0, the partial-credit
  levels, keep-out and hands-away rules, the demo's waypoints all reachable by IK).
- Debug: `TASK_TRACE=<file>` logs the reading, both iris offsets, the knob angles, the tool points and the grippers'
  opening every 10 calls. `PROOF=<knob>` (e.g. `PROOF=m1_side`) makes the demo turn one knob 0.47 turn on its own.

## Scene (positions from the midpoint between the arm bases: x forward, y to the left arm, heights above the bench top)

Beam height 200 mm. Every part stands on a Ø1" pedestal post held by a clamping fork bolted to a 400 x 460 x 12.7 mm
aluminium plate (tapped 1/4-20 at the six post positions) centred at (0.40, 0.00), the plate clamped to the bench.

| Item | Position | Product |
|---|---|---|
| Laser module, 650 nm, 1 mW, Class 2, collimated dot, Ø12 x 40 mm, fires along -x | exit at (0.56, +0.11) | Picotronic DBI650-1-3-FA(12x40), in a Thorlabs AD12NT adapter in an LMR1 lens mount |
| Mirror M1, normal at -45 deg (faces +x, -y) | (0.28, +0.11) | Thorlabs KM100 + Ø25.4 mm first-surface mirror |
| Mirror M2, normal at +45 deg (faces +x, +y) | (0.28, -0.11) | same |
| Iris 1, closed to its 0.8 mm stop | (0.40, -0.11) | Edmund Optics #53-914 mounted iris (30.8 mm OD, 0.8-12 mm) |
| Iris 2, closed to its 0.8 mm stop | (0.52, -0.11) | same |
| Photodiode in a black holder, behind iris 2 | (0.555, -0.11) | Vishay BPW34 |
| Multimeter, face up, µA range, reading the photodiode's short-circuit current | (0.45, -0.25) | any DMM with a µA range |

Each KM100's two adjuster knobs (Ø12 mm, 1/4-80 thread: 0.318 mm per turn, about 8 mrad of mirror tilt per turn on
the mount's lever, an estimate used in the simulation) face the robot from the back of
the mount, at the plate's corners 12.7 mm up-and-left and 12.7 mm down-and-right of the mirror's centre (seen from
behind): the upper knob tips the mirror up and down, the lower one turns it left and right. Each knob carries a white
paint line at its aligned position, matched by a line on the mount.

**Detector.** The BPW34 is wired straight to the multimeter's µA input (short-circuit photocurrent, linear in the
optical power, in the µA range through the 0.8 mm irises). The dark reading (laser blocked) is subtracted from every
reading.

**Before each trial (reset, about 3 minutes).** With every knob on its line and both irises at their 0.8 mm stop: block
the laser and note the dark reading; unblock and note the aligned reading P_ref. Then turn each knob from its line by the
seeded number of turns (signed: + is counter-clockwise seen from behind the mount), counting whole turns by the paint line
and setting the fraction against a printed 10-degree dial card held behind the knob. The beam then misses the near iris
and the meter reads under 1% of P_ref.

Seeded starting offsets (turns from the aligned line; `task.OFFSETS`):

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

- **0** if at any time any gripper or robot link touched the laser, either iris (or its lever), the photodiode or a
  mirror's face, or pushed or twisted a post or mount body (trial video; in simulation, a tool point inside a keep-out
  box around each), or if at the end a gripper is still within 5 cm of a knob.
- Otherwise by the photocurrent P (dark subtracted) as a fraction of P_ref: **1.0** at P >= 0.80 P_ref, **0.5** at
  P >= 0.40 P_ref, **0.25** at P >= 0.10 P_ref, **0** below.

Doing nothing scores 0 (every seed starts under 0.01 P_ref).

## How the simulation works (fidelity)

The beam is not rendered as light. Every step, `beam()` ray-traces it from the laser off both mirror planes (exact
reflection) to the two iris planes, from the four knob angles: one knob turn tilts its mirror 8 mrad about the
corresponding axis (estimate: the 1/4-80 thread's 0.318 mm per turn on a lever of about 40 mm). The reading is the Gaussian beam's transmission through both
0.8 mm irises, exp(-2 (r1² + r2²) / w²) with the module's 1/e² radius w = 1.5 mm, relative to the aligned reference. Red
spots on the irises (drawn 6 mm across so they can be seen) and a green bar on the meter display it; the beam from the
laser to M1 to M2 is drawn as a red line. The knobs are real revolute joints (0.03 N·m friction) that the grippers grip
and turn; the mounts and posts are fixed (bolted). The touch rules use tool-point keep-out boxes (laser, irises with
their levers, photodiode, mirror faces, each mount's body and post). Simplified: no beam divergence or clipping on the
mirror edges beyond a check that the beam lands on M2's face; the mirror plates do not visibly tilt (a few mrad).

## Demo

`demo()` (policy `yam_scripted`) reads the starting knob angles from the simulator (allowed for scripted demos) and
turns each back to zero: the left arm does M1's two knobs, the right arm M2's. Each knob is gripped from behind along
its axis and turned by rolling the wrist (up to about 230 degrees in one grasp; longer turns take a second grasp);
between knobs the wrist turns back and the hand returns to its start point. At the end both hands move clear and hold
for 6 s while the meter is read. Result: seed 0, `task_score` 1.0: reading 0.97 of the aligned reference, every knob within 0.02 turn of aligned; success held from about 31 s, the run ends at 36 s (726 control steps, a quarter of the run limit).

## Dream-1

Dream-1 (seed 0, same instruction, 3,000-step limit): `task_score` 0.0. Neither arm reached a mount: both drifted from their start poses toward the near bench edge over the run, no knob turned, and the meter read 0 throughout (`out/dream2.trace`).

## Safety

The laser is Class 2 (under 1 mW, visible): the blink reflex protects the eye. The beam runs horizontal at 200 mm above
the bench, below a standing person's eye level, and ends on the photodiode; no reflective tools are used near it.

## Real parts (lowest-cost honest build)

| Part | Qty | Unit price (USD) | Source |
|---|---|---|---|
| Thorlabs KM100 kinematic mirror mount, Ø1", two 1/4-80 adjuster knobs | 2 | 44.78 | https://www.thorlabs.com/thorproduct.cfm?partnumber=KM100 |
| Ø25.4 mm first-surface (front-surface aluminium) mirror | 2 | approx. 15 | any optics supplier |
| Picotronic DBI650-1-3-FA(12x40) dot laser module, 650 nm, 1 mW, Class 2 | 1 | approx. 50 (EUR 46.13 in the maker's catalogue) | https://www.picotronic.de |
| Thorlabs AD12NT adapter (Ø12 mm bore) in an LMR1 Ø1" lens mount | 1 | approx. 45 | thorlabs.com |
| 3 V supply for the module (2 x AA holder with switch) | 1 | approx. 5 | |
| Edmund Optics #53-914 mounted iris diaphragm, 0.8-12 mm | 2 | 63.50 | https://www.edmundoptics.com |
| Vishay BPW34 photodiode in a printed black holder | 1 | approx. 2 | digikey.com |
| Digital multimeter with a µA range | 1 | approx. 25 | |
| Ø1" pedestal post + clamping fork (e.g. Thorlabs RS-series pillar post + CF125) | 6 | approx. 30 | thorlabs.com |
| 6061 aluminium plate 400 x 460 x 12.7 mm, drilled and tapped 1/4-20 at the six post positions (drawing: the positions in the scene table), clamped to the bench | 1 | approx. 60 | any metal supplier |

Total about USD 610 (about USD 230 of it verified list prices: the mounts and the irises; the rest approximate).
Higher-end option: a Thorlabs 300 x 450 mm breadboard and protected-silver mirrors (PF10-03-P01) instead of the plate and
the first-surface mirrors, roughly USD 300 more.
