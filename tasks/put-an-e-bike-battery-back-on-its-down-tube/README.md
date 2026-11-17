# Put an e-bike's frame battery back on its down tube

An e-bike frame stands in a repair stand at the bench, its down tube rising 25 degrees toward the left arm, with the two
holders of a Bosch PowerPack Frame battery on top of it. The battery (338 x 93 x 81 mm, 3.0 kg) lies on its flank on
the bench. It has to go back the way the Bosch manual says: bottom end first, so that the contact block on its back
slides into the pocket of the lower holder, out of sight under the battery's own end, then swung down about that
pocket into the upper holder and pressed until the latch clicks.

What makes it hard: a 3 kg, 34 cm part held in two hands has to be turned so its back faces the tube and brought to
the lower holder within a few millimetres and a few degrees (the pocket gives 3 mm of play and lets the battery tilt
only about 7 degrees off the tube), with the contacts hidden; then it pivots about that blind joint and must be pressed
home against the latch spring without being pulled out of the pocket. A dropped battery scores 0.

## Run a trial

```
git clone https://github.com/kairosiann/yam-isaac-lab-arena ~/yam-isaac-lab-arena   # yam_arena embodiment
inspect-robots run --embodiment yam_arena --policy <policy> -E task=task.py \
    --instruction "Put the battery back on the e-bike." \
    --scorer task_score -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 --max-steps 2000
```

- Scripted demo: `--policy yam_scripted` (plays `demo()` in `task.py`).
- Run limit: 2,000 steps (100 s at 20 Hz). The demo finishes in about 600 steps (30 s).
- Runtime: about 4 to 6 minutes per trial on one GPU (Isaac Sim start-up included).
- Seeds: none; one fixed layout (`object_jitter=0`). With `TASK_TRACE=<file>` the task writes a JSON line every
  few steps (battery pose against home, plunger and bolt, the hands' tool points and openings).
- Offline checks (no simulator): `PYTHONPATH=~/yam-isaac-lab-arena uv run --python 3.12 --with numpy --with pytest
  python -m pytest test_task.py` (scoring on made-up states, holder geometry, demo reach).

## Scoring (`score()` in `task.py`)

Judged at the end of the trial, except a fall, which latches 0.

| Outcome | Score |
|---|---|
| The battery falls (drops out of the grippers onto the bench, the frame or the floor) before it has clicked in | 0, for good |
| Bottom end not home | 0 |
| Bottom end home in the lower holder: contact block in the pocket, end within 3 mm of home along the tube and 4 mm sideways, square to the tube within 5 degrees | 0.25 |
| ... and top end swung in to within 10 mm of the upper holder's seat (not clicked) | 0.5 |
| Clicked into the upper holder's latch, and still home at the end | 1.0 |

In the real trial: the fall and the click from the video; "home" by eye and a ruler at the bottom end (the end of
the battery against the lower holder) and at the top end (gap to the upper holder); clicked = the top end cannot be
lifted off the upper holder by hand after the trial. In the simulator a fall is the battery's centre of mass dropping
faster than 0.6 m/s between two control steps; clicked is the battery pressed to within 1.5 mm of home at its top end
(the latch plunger within 1 mm of flush) with its bottom end home, after which the latch bolt runs into the notch in
the battery's top end and holds it. The episode ends early once it has clicked and both hands are 5 cm off it.

## Scripted demo

Both hands grip the battery across its depth (top face and back face), near its arm-side flank, at 8 and 26 cm from
its bottom end (either side of its centre of mass); lift it, roll it a quarter turn so its back faces down, raise it
in front of the down tube, move it over the lower holder tilted 9 degrees, lower its bottom end onto the holder's plate
3 cm up the tube at 5 degrees, slide the contact block down into the pocket, swing the battery down onto the upper
holder, press (22 N per hand along the battery's back-face normal) until the latch clicks, open and back off.
Open loop on Cartesian waypoints with IK; the simulated arms compensate their own weight but not a load, so every
target is pre-deflected by the joint deflection the hand's share of the battery's weight (and, at the end, the
pressing force) will cause, K^-1 J^T w, with the arms' joint stiffness (80 N m/rad on joints 1-3, 10 on joints 4-6).
No teleporting and no changes to the simulation.

Result (1280 x 704): `outcome: succeeded`, `task_score: 1` - clicked in at 25 s, hands off and episode ended at 30 s.

## Dream-1

One run on the final task, 1280 x 704, 2,000 steps: `outcome: hit step limit`, `task_score: 0`. The battery never
moved. The right arm (on the right of the video) closed its fingers and hovered over the lower holder and the drive
unit (8-40 s and 60-70 s), dropped in front of the drive unit to the right of the battery and came within 6 cm of the
battery's bottom end (78 s), then went back toward its base; the left arm stayed near its base with its fingers closed
and never came within 15 cm of the battery.

## Real setup (one fixed layout)

Positions in cm from the midpoint between the arm bases: x forward, y toward the left arm, z above the bench top.

| Item | Real part | Where |
|---|---|---|
| Battery | Bosch PowerPack Frame 540 (BBP3553), 338 x 93 x 81 mm, 3.0 kg, black plastic casing; for trials a retired one (failed cells, BMS-locked) at its full weight | On its right flank on the bench, length along y from -17 (bottom end, right) to +17, its back (contact block side) toward the arms at x 16, top face at x 24 |
| Battery holders | Bosch PowerPack Frame holder kit for the smart system (plug-side lower holder, lock-side upper holder; EB1290000R, about EUR 4 per part), key not in the lock | On the down tube: lower holder just above the drive unit (its end wall at y -19, z 15), upper holder 34 cm up the tube (y 16 to 19, z 31 to 35) |
| Frame | A Bosch smart-system step-through e-bike frame with a down-tube PowerPack Frame mount (aluminium, 64 mm down tube), drive unit fitted, wheels, fork, cranks and pedals off | Upright at x 34, front toward the left arm; down tube rising 25 degrees from the drive unit (centre line y -23, z 9) to the head tube (y 42, z 39); drive unit 20 x 10 x 14 cm, 1 cm above the bench, crank axle at y -31, z 7; seat tube up to the stand's clamp at about y -47, z 62; rear stays past the bench's right edge |
| Stand | Bike repair stand (e.g. Park Tool PCS-10.3) behind the bench, clamping the seat post | Clamp at about x 34, y -47, z 62 |

Home, the battery's bottom back corner is at x 34, y -15, z 17. Nothing else is on the bench.

Reset: open the lock with its key (Bosch: the battery is then held by the safety catch), tilt the battery out of the
upper holder, pull it out of the lower holder, take the key out, and lay the battery back on its flank at the start
position. Safety: the frame is clamped; the battery is a retired one, can fall at most about 40 cm, onto a soft mat
laid on the bench under the frame; nobody reaches into the arms' workspace during a trial.

## Sources

- Bosch PowerPack Frame batteries (PowerPack 540 Frame: 338 x 93 x 81 mm, 3.0 kg):
  https://www.bosch-ebike.com/en/products/batteries/powerpack-frame
- Bosch eBike battery operating manual 0 275 007 3PX (2026-04-27), "Inserting and Removing the PowerPack Battery":
  key not inserted; place it so that its contacts are in the lower holder; twist it into the upper holder as far as
  possible until you hear it click into place. Product code BBP3553, 3.0 kg.
  https://www.e-bikeshop.co.uk/cdn/shop/files/Bosch-MY27-CompactTube-PowerTube-PowerPack-eBike-Battery-User-Manual.pdf
- Older Bosch manual (frame battery may be tilted up to 7 degrees toward the frame while on the lower holder):
  https://www.manualslib.com/manual/2776532/Bosch-Bdu280p.html?page=40
- Holder kit: https://www.ebike24.com/bosch-powerpack-frame-battery-holder-kit-smart-system/8455-004
- E-bike numbers: ZIV (Germany, 2024: 2.05 million e-bikes sold, 53 % of all bikes); CONEBI (Europe, 2023: 5.1 million
  e-bikes sold).

## Simulation notes

Rigid bodies throughout; tubes are cylinders, the drive unit a box, the battery a box with its contact block and a
latch notch. The exact shapes of the Bosch holder's pocket and the battery's contact block are not published, so they
are approximations sized like the real parts: the block is 50 x 20 x 20 mm, the pocket gives 3 mm of play each way and
lets the battery tilt about 7 degrees. The latch is a spring plunger under the battery's top end (15 N when up, 30 N
pressed flush) and a bolt that runs out of the upper holder into the battery's notch once the battery is pressed home
(a rule). Friction 0.6 on the frame, holders and battery. No third-party assets: every shape is a primitive defined
in `task.py`.

## Files

- `task.py`: scene, scoring, scripted demo. `ebkit.py`: the fixture kit (fixed and free articulated bodies, rotated
  frames, spring sliders, the latch flag). `test_task.py`: offline tests.
