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

Final state, from a top photo (the wall scales), a straightedge across the pins, and the trial video.

- **0** if either rack is not seated in its channel, or the fingers are off-centre.
  - Seated means: top flush with the pins' tops within 1.5 mm; against its wall within 1.5 mm; within 3° of level and
    4° of straight.
  - Off-centre means |s_white − s_orange| / 2 over 2.0 mm. One tooth off is always 4.7 mm; backlash allows 0.4 mm.
- **1.0**: seated and centred, and all of the following:
  - each rack travelled at least 15 mm apart while the orange rack was not touched. On video: only the white rack, or
    the pinion, was handled during the opening. In the simulation: no robot link touched the orange rack and no
    tool point was within 20 mm of it.
  - both fingers at least 15 mm open;
  - then released: racks still and every gripper 30 mm clear, held for 4 s.
- **0.5**: seated and centred otherwise.

In the simulation, every control step reads:
- the racks' poses in the housing's frame (the housing is free);
- the pinion's angle;
- the gripper tool points;
- PhysX contacts between every robot link and each part.

The record of coupled travel is cleared at each episode reset.

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
  - a rack lifted out, a tilted rack, both racks dragged by hand, and a gripper left near the jaws.
- Runtime per trial: about 5 minutes wall clock for the scripted demo (559 steps) on one shared GPU, plus about 1 to 2 minutes of Isaac Sim start-up; a full 1,200-step trial takes about 10 minutes.
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

Result (task file sha256 945338f5ebce, 1280 x 704):
- task_score 1.0, ended as a success at step 559 of 1,200 (28 s simulated, 276 s wall clock).
- The orange rack was seated at step 270, at the centring tooth: d = −0.09 mm.
- Coupled opening: 18.96 mm of travel by each rack with the orange rack untouched.
- At the end: fingers 29.3 and 29.05 mm open, d = 0.12 mm, both racks seated (0.0° tilt), housing unmoved.
- The arms parked clear for the last 4 s, so the final frames show the released, centred, open jaws.

## Dream-1

One run on the final files (same task.py, sha256 945338f5ebce, the final instruction, 1280 x 704, 1,200 steps):
task_score 0.0, hit the step limit.
- Both arms rose to about 25 cm above the table: the left arm on the left of the video, the right on the right.
- From about 20 s on, they hovered with their grippers turned toward the parts, slowly drifting toward the robot's
  side.
- They never came down: no gripper came closer than 28 cm to the white rack or 32 cm to the orange rack.
- Nothing was touched; the orange rack stayed where it started on the table.

## How it is modelled

- `task.py`: scene, `score()`, `success()`, `demo()`.
- `gripkit.py`:
  - builds the housing as one free PhysX articulation, with the pinion on a revolute joint;
  - builds each rack as a free rigid body;
  - reports poses, joint angle and robot contacts.
- Geometry and masses follow the printed parts. There are no rules: seating, meshing and the coupled drive are all
  contacts.
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
