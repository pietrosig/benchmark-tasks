# Assemble a rack-and-pinion parallel gripper, seating the second rack at the tooth that centres its jaws

Takes up the idea of TK-JUWHH6 after its 48 h hold lapsed (review 2026-11-16 06:23 UTC, no revision).

A 3D-printed parallel-jaw gripper of the common kind lies on the bench, half assembled. One pinion sits between two
channels, and each channel takes a rack that carries a finger. The white rack is in the near channel, one tooth
(9.4 mm) open, and the pinion has turned with it. The orange rack lies loose beside the housing. The robot seats the
orange rack on the pinion at the tooth that centres the jaws: its finger as far from its stop as the white one.
Then, with its hands off the orange rack, it pulls the white finger at least 15 mm further open. The pinion drives the orange
rack the other way, which proves the mesh.

What is physically hard:
- **Blind meshing on a free pinion.** The orange rack is lowered onto eight 3 mm pins. Its 3.4 mm tooth gaps must
  fall over them, with the pinion free to turn and the white rack free to slide.
- **Choosing the tooth from the other rack's position.** The right tooth is the one where the orange finger's opening
  equals the white finger's (here 9.4 mm, one tooth). Landing one tooth off still meshes, but the fingers then sit
  4.7 mm off-centre for good.
- **Leaving the rest undisturbed.** Pressing the rack home must not knock the pinion or the white rack.

## Instruction

> Seat the loose orange rack on the pinion so the gripper's two fingers sit evenly about its centre, then pull the white finger at least 15 mm further open, without touching the orange rack, and let go.

## Real setup

Positions in mm from the midpoint between the two arm bases: x forward, y to the left. Heights are above the table
top. Nothing is fixed to the table.

| Part | What | Size, mass, material | Where |
|---|---|---|---|
| Housing | printed PETG floor and walls (cad/gripper.scad, cad/housing.stl) glued on a steel flat bar 160 x 50 x 6 mm | floor 160 x 52 mm; walls 12 mm tall, inner faces 45 mm apart; dark grey; a 5 mm rail under each rack; stop blocks at y +40 to +44 (white channel) and -44 to -40 (orange channel); a yellow 1 mm scale along each wall top, zero at the closed finger plate; 460 g in all | free on the table (nothing fixed), centre (240, 0), long side along y, on four tape corner marks |
| Pinion | printed disc (27 mm, 4 mm thick) and hub (cad/pinion_disc_and_hub.stl), eight hardened dowel pins ISO 8734 / DIN 6325 m6, 3 x 16 mm (Zoro Select M390800300016, 100-pack) pressed 3.5 mm deep into the disc on a 24 mm circle, 12.5 mm standing, turning on a shoulder screw ISO 7379, 5 mm shoulder, M4 (McMaster-Carr 92981A series) | pitch 9.42 mm (module 3), 10 g, turns under 0.002 N m | centre of the housing |
| White rack | printed PETG (cad/rack_near.stl) | 88 mm long, 8 mm body plus 2.8 mm teeth, 12 mm tall; tooth gaps 3.4 mm (0.4 mm backlash on a 3 mm pin); rounded crests, lead-in feet; finger plate 7.5 x 6 x 30 mm at its outer end; finger-mount boss 7 x 6 x 19.5 mm on top (real fingers bolt to it, M3); 13 g | near channel (pitch line x 228), one tooth (9.4 mm) open from its stop |
| Orange rack | printed PETG (cad/rack_far.stl), the same rack mirrored | as above, 13 g | loose, upright on the table, pitch line at x 252, middle at y 140, teeth toward -x |

The opening s of a rack is its finger plate's travel from its stop, read on the 1 mm scale along that channel's wall
top. The fingers are centred when s_white = s_orange.

Print settings: PETG, 0.2 mm layers, 4 perimeters, 40 % gyroid infill. Racks print lying on their side; no supports.
Fit checks after printing:
- each rack slides in its channel under 20 gf;
- with the white rack on its stop, a pin sits in a tooth gap of each rack;
- backlash measured at a finger is at most 0.5 mm.

## Scoring

Final state, from a top photo (the wall scales), a straightedge across the pins, a hand check of the mesh, and the
trial video.

- **0** unless both racks are in mesh with the pinion and the fingers are centred.
  - **Seated** in its own channel: top flush with the pins' tops within 1.5 mm; against its wall within 1.5 mm; within
    3° of level and 4° of straight.
  - **Over the pinion:** the pinion's axis lies between the rack's first and last tooth, so the pin at the mesh point
    has a tooth on each side of it. In the photo, at least two of the rack's teeth are across the pinion's pins.
  - **In mesh:** the pin at the mesh point sits in one of the rack's tooth gaps. A rack lying on the pin tips is not
    seated, and a rack slid out past the pinion does not overlap it.
  - **Centred:** |s_white − s_orange| / 2 at most 2.0 mm. One tooth off is always 4.7 mm; backlash allows 0.4 mm.
- **1.0**: in mesh and centred, and all of the following:
  - each rack travelled at least 15 mm apart while the orange rack was not touched. On video: only the white rack, or
    the pinion, was handled during the opening. In the simulation: no robot link touched the orange rack and no tool
    point was within 20 mm of it.
  - both fingers at least 15 mm open;
  - then released: racks still and every gripper 30 mm clear, held for the last 4 s.
- **0.5**: in mesh and centred otherwise.

**Real-trial mesh check**, after the end photo:
1. The tester pushes the white finger 5 mm back toward its stop with one finger (outward instead, if it is less than
   5 mm from its stop).
2. On the wall scales, the orange finger must move 5 mm the same way relative to its own stop, within 2 mm (backlash
   up to 0.5 mm, plus reading 1 mm scales by eye). If it doesn't, the trial scores 0. Racks not in mesh don't move
   together, and a rack out past the pinion can't be driven.
3. The tester then reads both scales once more and checks the fingers are still even.

**How the 4 s hold is enforced:**
- In yam_arena, the trial framework calls the task's `success()` every control step. It counts the consecutive steps
  for which `success()` is true, and ends the episode with termination_reason "success" once the count reaches
  `SUCCESS_HOLD_S` × 20 = 80 steps (`yam_arena/inspect_embodiment.py`, `_task_step`).
- The `task_score` scorer then reports 1.0 (`yam_arena/scoring.py`).
- A run that reaches its step limit first is scored by the last step's `score()`. So `score()` in task.py counts the
  same hold itself: it returns 1.0 only after the full-score state has held for 80 consecutive control steps, and
  0.5 before that.
- Either way, a 1.0 requires the full 4 s. Tests: `test_full_score_needs_80_steps_of_hold`,
  `test_hold_restarts_if_interrupted`, `test_success_flags_the_state_and_yam_arena_counts_the_same_hold`.

In the simulation, every control step reads:
- the racks' poses in the housing's frame (the housing is free);
- the pinion's angle;
- the gripper tool points;
- PhysX contacts between every robot link and each part.

The record of coupled travel, and of the hold, is cleared at every environment reset: gripkit's reset hook calls the
task's `_reset_record`, and the record is also cleared if the step counter ever goes backwards.

### Earlier feedback on TK-97PN32 and how it is covered

1. **Placing both racks evenly on the rails beyond the pinion scored 0.5.** Now every tier needs both racks over the
   pinion and in mesh with it, and the coupled-travel record only counts while both are in mesh. Tests:
   - `test_both_racks_symmetric_beyond_the_pinion_score_zero` (racks 45 to 60 mm open, off the pinion);
   - `test_symmetric_beyond_the_pinion_held_still_never_scores`;
   - `test_one_rack_meshed_and_one_beyond_scores_zero`;
   - `test_rack_between_teeth_phases_is_not_meshed`;
   - `test_pinion_turned_away_from_both_racks_is_not_meshed`;
   - `test_meshed_at_the_wrong_tooth_scores_zero`;
   - `test_meshed_and_even_at_many_openings`;
   - `test_mesh_holds_at_both_backlash_extremes`.
2. **A practical real-trial check for the mesh.** The end photo must show the rack's teeth across the pins, and the
   hand check must pass: 5 mm on the white finger moves the orange finger 5 mm, within 2 mm, or the trial scores 0.
3. **Verify the four-second hold.** See "How the 4 s hold is enforced" above.

### Earlier feedback on TK-JUWHH6 and how it is covered

1. "A robot can remove the near rack and position it alongside the housing": both racks must be seated in their own
   channels. Tests: `test_near_rack_lifted_out_beside_the_housing_scores_zero`, `test_unseated_rack_scores_zero`,
   `test_tilted_rack_scores_zero`.
2. "Demonstrate that actuating one rack drives the other": credit only for travel with the orange rack untouched.
   Tests: `test_opened_by_the_white_rack_alone_scores_one`, `test_both_racks_dragged_by_hand_scores_half`,
   `test_dragging_the_orange_rack_does_not_count`, `test_open_but_not_far_enough`.
3. "A wrong teeth index scores 0": |d| over 2.0 mm scores 0, where one tooth is 4.7 mm. Tests:
   - `test_wrong_tooth_scores_zero` (±1 and 2 teeth);
   - `test_right_tooth_passes_at_backlash_extremes`;
   - `test_backlash_is_under_half_a_millimetre`;
   - `test_mesh_has_clearance_at_every_angle`.

## Run and score a trial

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=1280 -E cam_height=704 --instruction "$(cat instruction.txt)" \
  --scorer task_score --max-steps 1200 --store-frames --log-dir logs/demo
```

- For Dream-1 use `--policy dream`.
- `TASK_TRACE=trace.jsonl` writes, every 10 steps, what the score is built from.
- Offline tests, no simulator: `python -m pytest test_task.py` with yam_arena on `PYTHONPATH`. They cover:
  - the mesh clearance at every pinion angle;
  - backlash at most 0.5 mm;
  - the right tooth passing at both backlash extremes, and ±1 and 2 teeth off scoring 0;
  - a rack lifted out, a tilted rack, both racks dragged by hand, and a gripper left near the jaws;
  - racks placed evenly beyond the pinion, one rack meshed and one not, a rack between tooth phases, and a pinion
    turned away from both racks (all 0);
  - the 4 s hold (79 steps score 0.5, 80 score 1.0, and the hold restarts if interrupted).
  There are 33 tests.
- Runtime per trial: about 5 to 6 minutes wall clock for the scripted demo (559 steps) on one shared GPU, plus about 1 to 2 minutes of Isaac Sim start-up; a full 1,200-step trial takes about 10 minutes.
- Seeds: none; the layout is fixed (`-E object_jitter=0`) and the scripted demo is open loop.

## Scripted demo

`demo()` in `task.py`: fixed waypoints (open loop), played by `yam_scripted`.

1. Left arm (on the left of the video):
   - pinches the orange rack's boss on the bench (pads 30 mm open, closing along the rack);
   - lifts it and carries it over the far channel, at the white rack's opening (one tooth, 9.4 mm, where the fingers
     sit evenly);
   - lowers it in steps until its teeth meet the pins, presses it 2 mm further, lets go and parks to the left.
2. Right arm, once the left is clear:
   - pinches the white rack's boss;
   - pulls it 21 mm further open; the pinion drives the orange rack the other way, with no hand near it;
   - lets go and parks to the right.

Result (task file sha256 1798dafaf4bd, 1280 x 704):
- task_score 1.0, ended as a success at step 559 of 1,200: after the full-score state had held for 80 control steps (4 s).
- The orange rack was seated and in mesh from about step 260 of the episode, at the centring tooth: d = −0.07 mm.
  The trace's step counter runs about 10 ahead of the episode's.
- Coupled opening: 18.96 mm of travel by each rack with the orange rack untouched.
- At the end: fingers 29.3 and 29.05 mm open, d = 0.12 mm, both racks over the pinion and in mesh, housing unmoved.
- The arms parked clear for the last 4 s.

## Dream-1

One run on the final files (same task.py, sha256 1798dafaf4bd, the same instruction, 1280 x 704, 1,200 steps):
task_score 0.0, hit the step limit (key_in_outputs 0). From the trace and the six sampled frames:
- **Left arm** (on the left of the video): rose to 24–34 cm above the table (median 28 cm) and hovered with its
  gripper turned toward the parts. Its closest approach was 25 cm to the orange rack, at about 22 s.
- **Right arm** (on the right of the video): rose to 20–29 cm above the table (median 26 cm) and hovered the same
  way. Its closest approach was 24 cm to the white rack, at about 23 s.
- Both drifted slowly toward the robot's side over the run.
- No tool point came closer than about 23 cm to any part, and nothing was touched. The orange rack stayed where it
  started on the table, so it was never in mesh.

## How it is modelled

- `task.py`: scene, `score()`, `success()`, `demo()`.
- `gripkit.py`:
  - builds the housing as one free PhysX articulation, with the pinion on a revolute joint;
  - builds each rack as a free rigid body;
  - reports poses, joint angle and robot contacts.
- Geometry and masses follow the printed parts. There are no rules: seating, meshing and the coupled drive are all
  contacts.
- The scorer reads the mesh from the racks' poses and the pinion's angle:
  - the pinion's axis between each rack's first and last tooth;
  - the pin nearest the mesh point within 0.7 mm of a tooth-gap centre (the gap's 0.2 mm clearance plus 0.5 mm).
- The gripper's cover and actuator (servo or cylinder) are omitted: it is assembled bare.
- Simplifications:
  - the teeth's rounded crests and lead-in feet are stepped boxes (the printed parts have true chamfers, gripper.scad);
  - the pins' rounded ends are spheres.
- Third-party assets: none.

## Sources

- Festo HPPF parallel grippers (datasheets, e.g. HPPF-8-32-A): the jaws are driven by a rack and pinion, with
  positively guided synchronous jaw motion. This is the same principle as this task's gripper.
- MakerWorld model 2188993, "Rack And Pinion Parallel Grabber": its assembly notes slide the two racks into the side
  slots, teeth inward, and place the pinion in the centre so that it meshes with both.
- US patent 6,505,991, "Self-centering shaft adapter": a pinion between two racks drives two jaws equal distances in
  opposite directions.
