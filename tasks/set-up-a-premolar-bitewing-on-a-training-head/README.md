# Set up a premolar bitewing on a dental training head

A dental training head is clamped on the bench, its mouth open. Beside it, a bitewing holder lies on a tray, assembled:
a Rinn XCP-ORA arm and aiming ring, a red XCP bitewing bite block and a size 2 phosphor plate (31 x 41 mm) in its
barrier envelope. The robot sets up a bitewing of the patient's right premolars. The plate has to stand between the
tongue space and the lower right premolars, parallel to them, its front edge at the canine, with the bite tab resting
on the lower premolars. Then the jaw is closed onto the tab and the holder let go. No X-ray is taken. The aiming ring,
which the arm holds on the beam axis outside the cheek, then shows where the cone would go.

Why it matters: dental assistants (O*NET 31-9091, 388,600 US workers: "Expose dental diagnostic x-rays") and dental
hygienists (29-1292, 225,300: "Expose and develop x-ray film") take bitewings at routine check-ups, several sets a
day. The skill is placing the receptor. A flat 41 x 31 mm plate has to reach an exact pose inside a true-size mouth,
mostly out of the cameras' view: past the front teeth through a 48 mm opening, behind the premolars, parallel to
them, its front edge at the canine. Too far forward or turned off the row, the image overlaps or misses teeth and has
to be retaken. On a training head the operator closes the jaw. On a patient the patient bites while the operator
holds the instrument still. Either way the plate must not move while the jaw closes on its tab. Bitewing placement is taught and checked on
phantom heads like this one, with the instructor or student closing the mandible. The skill scored here, placing the
plate and keeping it undisturbed through the closure, is the one used on patients.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the head, tray and holder, `score()`, `success()`, `distance()` and the scripted `demo()` |
| `head.py` | Geometry of the training head and the holder, built from their measurements (pure NumPy) |
| `mouthkit.py` | Articulated fixtures for yam_arena task files (hinged parts with joint friction, mesh shapes, free bodies) |
| `test_task.py` | Offline checks (no GPU): the scoring rules on made-up states, and the demo's reach |
| `instruction.txt` | The instruction |

`mouthkit.py` hooks `yam_arena.scene` when it is imported, without modifying yam_arena. The task runs with the
standard `yam_arena` embodiment and the `task_score` scorer. No third-party meshes or scans are used: every shape is
generated in `head.py`.

## Run

    inspect-robots run --embodiment yam_arena --policy <policy> -E task=task.py \
      --instruction "$(cat instruction.txt)" --scorer task_score \
      -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 --max-steps 1800

- Scripted demo: `--policy yam_scripted`. About 48 s (964 steps), well inside the 1,800-step run limit.
- Seeds: `TASK_SEED=0,1,2` puts the tray's centre at x 24.5, 25.5 or 26.5 (the holder moves with it; default 0).
- Runtime: about 4 minutes per trial on one GPU, including start-up.
- Offline tests: `PYTHONPATH=<yam-isaac-lab-arena> uv run --python 3.12 --with numpy --with pytest python -m pytest test_task.py`

## Scene (sim and real)

Positions are in cm from the midpoint between the arm bases (x forward, y toward the left arm), heights above the bench.

- **Training head**: ANNWAH mouth simulator phantom head, "fix on table" model (Amazon ASIN B09NB9FSYQ, about US$115
  to 160). It has a silicone rubber face mask over a metal head form, a 28-tooth resin adult typodont (Nissin/Kilgore
  type) in aluminium jaws, and a lower jaw on a hinged articulator. It has no tongue. It is clamped to the bench by its
  table clamp, and its post's ball joint is 3 cm above the bench at x 37.5, y 0. It faces the robot and is reclined
  25 degrees (chin up) from upright, as on a dental chair. With a ring-aiming holder the head's tilt doesn't change the
  image, because the cone follows the ring. The head spans x 29.8 to 51.4 and y -9.1 to 9.1; its top is at x 42, height 30.6.
- **Jaw**: the hinge axis runs along y at x 40.9, height 15.8. The listing doesn't describe the jaw mechanism.
  Phantom-head articulators hold the mandible with a hinge or locking screw: the KaVo phantom-head patent US4944680A
  pivots the lower jaw against a locking mechanism, and the Paradigm (ADEM) phantom sets its opening with a rear bar.
  The task uses it as a friction hinge. If a unit has a latch, release it, and if it has a spring, back it off. Then,
  before each trial, set the hinge screw
  so the jaw stays at any opening under its own weight and under the holder's: about 0.45 N m to start it moving,
  which is about 5 N lifting under the chin (check with a spring scale). It must stay open when let go at its
  30 degree stop, and stay shut on the holder's tab with both grippers clear. There is no tug test: the bite only has
  to keep the holder in place under its own weight. The jaw starts open 30 degrees: 48 mm between the incisal edges.
- **Mask**: the mouth opening is 52 mm wide. The lower incisal edges are at x 31.5, height 16.8 when the jaw is shut.
  The patient's right is the robot's left (+y).
- **Holder**: assembled, 16 g (plate in envelope 3 g, bite block 2 g, arm 6 g, ring 5 g). It is a Rinn XCP-ORA One
  Ring & Arm (Dentsply Sirona REF 550771), a red XCP bitewing bite block and a size 2 phosphor plate (31 x 41 x 1.6 mm,
  in a barrier envelope 2 mm thick).
  - The ring is 72 mm across. It is slid along the arm's straight leg to 75 mm out from the plate on the beam axis.
  - The leg runs on 45 mm past the ring, as the arm does when the ring is pushed toward the face.
  - The robot pinches the leg 26 mm outside the ring, where a person holds the arm. Its fingers close across the leg
    and approach 30 degrees off its line, so they stay clear of the ring and the cheek.
  - The arm's bends are modelled as straight 6 to 7 mm bars at the real positions.
- **Tray**: a white plastic instrument tray, 20 x 13 x 0.8 cm, centred at x 24.5, y 26.2. The holder lies plate
  down on it, tipped 20 degrees onto the plate's long edge.
  - The plate's centre is at x 26.3, y 26.7, height 1.7.
  - The plate's long edge, bite-block end first, points toward the robot, 9 degrees to the robot's right of straight
    back.
  - The ring is above the plate. The arm's leg sticks out toward the robot, its end at x 15.6, y 24.9, height 10.9.
- Nothing else is on the bench. Both arms start at the ready pose with the grippers open.

## The target and the score

The target pose of the plate is defined from the lower teeth (in `head.target_frame()`):
- The tab's underside rests on the lower right premolars' cusps, so the plate's centre is 1.5 mm above the biting
  plane.
- The plate is parallel to the line along the lower right premolars' lingual faces, 7 mm lingual of it.
- Its front edge is level with the distal half of the lower right canine.

The score reads the plate's pose in the lower jaw's frame: u along the row, v up from the biting plane, d the plate
face's distance lingual of the premolars, h the horizontal angulation, t the tilt from upright.

- **1.0**: the plate is within 4 mm along the row, 3 mm vertically and 2 to 12 mm lingual of the premolars, with h
  and t each 10 degrees or less. The jaw is closed on the tab (jaw angle within 3.7 degrees of shut: an incisal gap of
  6 mm or less; the tab stops it at 2.25 degrees, about 4 mm). Both grippers' tool points are 3 cm or more from every
  part of the holder. The holder has been still for the last 1 s: on each of the last 20 steps (50 ms) its plate
  centre and arm end each moved under 1 mm, and it turned under 0.5 degrees.
- **0.75**: the same, with only the loose band met (8 mm, 6 mm, 0 to 16 mm, 20 degrees). The image would be
  readable, with overlap or a missed canine.
- **0.5**: within the full-score band, but the jaw is still open or a gripper still holds the holder.
- **0.25**: the tab rests on the lower right premolars or first molar, anywhere within 12 mm along the row and 6 mm
  vertically, the plate upright within 20 degrees.
- **0**: anything else, including nothing done, the left side, or the holder outside the mouth or dropped.

It also scores 0 for the rest of the run if a gripper's fingertips (its pad tips) pass the front of the lips into the
mouth opening: a finger in the patient's mouth. The score is read at the end of the run; the episode
does not stop early.

### Scoring a real trial

The plate is hidden in the closed mouth, but the XCP arm is rigid. So the ring's pose gives the plate's pose, and a
ruler, a square and a photo are enough.

1. **Calibrate once.** An operator places the holder by hand at the target and closes the jaw. Mount a phone on a
   clamp tripod (for example a Joby GripTight with a bench clamp) at a marked spot on the bench edge, on the beam
   axis 30 cm out, looking along the axis through the ring. Photograph the ring there, with a
   printed overlay of its outline and circles offset 3, 4, 6 and 8 mm (the full and loose tolerances along the row
   and vertically). A camera on the axis has no parallax across it. Mark on the base the line the ring's face lies
   along (for h) and the ring face's distance from a block beside the head along the axis (for d: +/-5 mm is the
   full band).
2. **After a trial**, take the same phone photo, read the ring centre's offset against the circles, the ring face's
   angle against the base line with a protractor (h), and its tilt from vertical with a square (t).
3. **Jaw closed:** measure the incisal gap with a ruler: 6 mm or less.
4. **Plate still in its slot:** a pen line is drawn on the plate's envelope at the slot's edge before the trial. After
   the photo, open the jaw and check the line is still at the slot's edge (+/-1 mm), or look in with a mouth mirror.
   A plate pushed along its slot is scored at its own measured position (the ring no longer stands for it).
5. **Let go, still, fingers:** judged from the trial video. The holder is let go and doesn't move in the last
   second, and no finger passes the front of the lips into the mouth.

## Scripted demo

The demo uses waypoints computed from the scene, IK on the arm's kinematics, and no teleporting.
- The left hand pinches the arm's straight leg outside the ring, lifts the holder and turns it upright. It carries the plate past the
  front teeth with its lower edge above the lower incisors, takes it back over the premolars, and lowers the tab onto
  them.
- The right hand comes in from the patient's left, closed fingers under the chin, and lifts the chin along its arc.
  The left hand carries the holder up with the lower jaw on the same schedule, leaning the tab 1 mm onto the teeth.
- The right hand presses the jaw shut on the tab, eases off (the hinge's friction keeps the jaw shut) and leaves.
  Only then does the left hand stop leaning, let go and back off along the arm.

## Fidelity

Rigid-body simulation (PhysX, 240 Hz physics, 20 Hz control).
- The typodont crowns, gums, palate and mask are rigid convex shapes at true size, built from the head's and a
  standard adult typodont's measurements, not scanned. The face mask is drawn as one smooth surface. Its colliders
  are the lips, the inner cheeks and the chin.
- The silicone lips and cheeks are rigid walls at their stretched positions, so they don't give.
- There is no tongue, as on the real head, so the open space lingual of the teeth matches the real trial.
- An operatory lamp in front of and above the face lights the mouth, as on a dental chair. It is lighting only: in real trials any room or desk lamp aimed at the mouth will do, and it is not scored.
- The jaw hinge has static and moving friction (0.45 and 0.38 N m), its friction screw's setting.
- No X-ray is simulated: the score reads the plate's pose, as the ring calibration does in a real trial.

## Results

| Run | Score (task_score) | What happened |
|---|---|---|
| Scripted demo (`yam_scripted`, seed 0) | 1.0 | At the end: plate 0.4 mm back along the row, 0.1 mm low and 5.7 mm lingual of the premolars; horizontal angulation 0.0 deg, tilt 1.6 deg; jaw closed on the tab (2.2 deg, incisal gap about 4 mm); both hands parked; holder still; no fingertip past the lips. 48 s. |
| Dream-1 (seed 0, 1,800 steps) | 0.0 | The left arm (on the left of the video) moved over the holder on the tray from about 10 s to the end. At 70 s its fingers came within 4 mm, and it nudged the holder 0.5 mm, but it never lifted it. The right arm stayed near its base, and the jaw stayed open. |

The demo was run on seed 0. Seeds 1 and 2 only move the tray and holder 1 and 2 cm along x; their reach is checked offline
in `test_task.py`.

## Parts

| Part | Source |
|---|---|
| ANNWAH mouth simulator phantom head, fix-on-table, typodont 28 teeth | https://www.amazon.com/dp/B09NB9FSYQ |
| Rinn XCP-ORA One Ring & Arm, REF 550771 | https://dmi.co.uk/products/xcp-0ra-single-pack-one-ring-and-arm-dent-550771.html |
| Rinn XCP bitewing bite blocks (red) | https://shop.benco.com/Product/1132-641/xcp-bite-wing-instrument-kit-red-2-horizontal-size |
| Size 2 phosphor storage plate and barrier envelopes (any PSP system) | dental supply |
| Plastic instrument tray, 20 x 13 cm | dental supply |
