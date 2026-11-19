# Cut three bites off a sausage with a knife and fork: tines pin the round, slick link while the knife saws through

A cooked sausage lies across a dinner plate. The robot takes the fork in its left hand and the serrated table knife in
its right, pins the link with the fork's tines and saws three bite-size pieces (1.5 to 2.5 cm) off its right end, then
lays the knife and fork down on the plate. The link is round and slick on the glazed plate: a knife alone drags and
rolls it, so the fork's tines must be in it while each cut is made, and the knife has to saw through to the plate.

Files: `task.py` (the yam_arena task file: scene, rules, scoring, scripted demo), `foodkit.py` (multi-shape rigid
bodies, runtime joints, force and contact helpers; imported by `task.py`, it hooks yam_arena without changing it),
`test_task.py` (offline checks, no simulator).

## Run and score a trial

```bash
# Dream-1 (or any policy)
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=1280 -E cam_height=704 --scorer task_score --max-steps 2400 --store-frames --log-dir logs/dream \
  --instruction "Cut three bite-size pieces, 1.5 to 2.5 cm long, off the sausage with the knife and fork, keeping everything on the plate, then lay the knife and fork down."
# the scripted demo: same command with --policy yam_scripted
```

`-E object_jitter=0` is required: every trial starts from the same layout (no seeds). A trial takes 3 to 6 minutes on
one GPU. `task_score` gives 1.0 once `success()` has held for 1 s, otherwise `score()` at the end (0, 0.25, 0.5,
0.75 or 1.0). With `TASK_TRACE=<file>` the task file writes a JSON line every 5 control steps (cuts, pieces, the
fork's hold, contacts, score).

Offline checks: `python -m pytest -q test_task.py` (scoring logic, layout, the demo's reach).

## Scoring

- **0 for the trial** if a gripper ever touches the sausage (PhysX contact between any link of either arm and the
  food) or any of it leaves the plate (a slab's middle more than 12 cm from the plate's centre, or below its floor).
- **A piece counts** when it has been cut fully off, is 2 slabs (2 cm; a cut 1.5 to 2.5 cm from an end lands on the
  2 cm line, see below), and the fork's tines were in the sausage when the cut that freed it went through.
- **0.25 per piece**, at most 0.75. **1.0**: three pieces, and the knife and the fork both let go, lying on the plate
  (middle within 13 cm of the plate's centre, highest point at most 5 cm above its floor, the fork not in the food),
  both grippers open and at least 3 cm from them; held for 1 s with everything still.
- Real trial: watch the video for gripper contact with the food, food leaving the plate, and the fork in the food
  during each cut; after the trial lift each piece with the fork (it must come away alone) and measure it with a ruler
  (1.5 to 2.5 cm); check the utensils lie on the plate and the grippers are open and clear.

## How it is modelled (estimates, to be checked on real food)

- **Plate**: kinematic (the non-slip mat holds it), floor plus a ring of rim panels; glazed: friction 0.3/0.25.
- **Sausage**: 13 rigid slabs, each 1 cm long and 3 cm across (6.2 g, a 1.2 cm flat underneath so it rests as a real
  link does), joined by fixed joints. Cuts happen only at the joints between slabs: the blade's plane is matched to the
  nearest joint within 5 mm, so a cut's position is resolved to 1 cm (a cut 1.5 to 2.5 cm from an end frees a 2 cm
  piece). Friction 0.5/0.45 (a cooked skin on glaze: about 0.4 combined).
- **Fork tines** (rule): the fork does not collide with the food. Pushed onto the skin, each tine tip's indent is
  resisted at 800 N/m along the skin's normal (plus at most 0.15 x that along the skin) until 2.5 N: then the tines go
  in, if they point within 55 degrees of the skin's normal (otherwise they skid, pushing at most 1 N). Once in, a D6
  joint holds the fork and that slab as they are, except along the tines: drawn out against a 3 N hold, or levered 35
  degrees over, the tines tear out; they cannot bite again for 1 s after that.
- **Knife edge** (rule): the knife does not collide with the food. The uncut food under the edge pushes it back at
  600 N/m (applied at the hand, so it does not turn the knife in the grip), at most 8 N; while the edge presses at least
  1 N, sawing along the blade deepens the cut by 0.3 mm per mm sawn; pressed past 8 N it goes through without sawing.
  The edge drags the food along with the stroke (12 N per m/s of blade speed, at most 1.2 N) at the top of the skin, so
  an unpinned link rolls or skids. The edge stays in the groove it first presses into until lifted out; a new cut can
  only start from the top of the skin. A cut is through when the edge is within 4 mm of the plate under the food.
  When a cut goes through, the two cut faces are pushed apart (0.07 N for 0.15 s, a few mm to about 1 cm), as the
  blade left in the kerf parts them, so a freed piece shows as separate.
- Only a knife in a hand cuts (a gripper within 3 cm of its handle), moved at under 0.5 m/s.
- Both forces are estimates: published puncture forces for cooked sausage casings with a fork, and the force to saw
  one with a serrated table knife, were not found; measure both on the specified link and utensils (a kitchen scale
  under the plate) before real trials.

## Demo

The left hand picks the fork up across its handle, tilts it (handle 40 degrees above horizontal, tines about 30
degrees off vertical), lowers the tines onto the sausage 10.5 cm from its right end and pushes them in. The right hand
lifts the knife from its rest by the handle's back end and, with the fork pinning the link, saws three times
(strokes of +-2.5 cm along the blade, lowering it as the cut deepens, never into the plate) at 2, 4 and 6 cm from the
right end. Then the left hand levers the fork 40 degrees over until the tines come out and lifts it clear and lays it across
the plate's far side; the right hand turns the knife and lays it along the plate's near rim, both clear of the food;
both let go and move back out of the way. The fork stays in the same place for all three cuts.

## Results

- Scripted demo: 1.0 (`task_score: 1`, success at 48.75 s; three 2 cm pieces cut off the right end with the fork in the
  link for every cut, the fork levered out and laid across the far side, the knife along the near rim, both let go). Video: `demo.mp4`.
- Dream-1: 0.0 at the 2,400-step limit. The left arm rose and drifted back toward its base; the right arm closed its
  gripper and from about 60 s hovered 4-5 cm above the table beside the knife's handle. Neither touched the fork, the
  knife or the sausage.

## Bypass checks

Three were run in the simulator on the final `task.py` with test hooks in `demo()` (`BYPASS=nofork|press|grab`; the
scene and scoring are unchanged); the others follow from the rules above.

| Try | Result |
| --- | --- |
| Do nothing | 0 (no cut) |
| Knife alone, no fork (`nofork`) | the first strokes dragged and rolled the link off the plate (cut 15 mm deep, no piece): 0. Pieces cut with no fork in the food never count |
| Grip the sausage itself to hold it (`grab`) | contact latched at once (both finger tips); the knife then cut three pieces, still 0 |
| Press straight down, no sawing (`press`) | the arm stalled under the 8 N the food needs; the deepest cut stopped at 16 of 30 mm, no piece: 0 |
| Fork 4-8 cm from the cut (the demo) | counts: the rigid-enough link is held anywhere along the rest |
| Knife against the plate's rim to trap the link | without the fork in the food no piece counts |
| Cut, then push pieces back together | a freed piece stays freed (cuts are not undone); it still counts if it is on the plate |
| Piece pushed off the plate and put back | the leave latched 0 |
| Fork or knife held near the food without touching | nothing happens: tines need 2.5 N into the skin, the edge 1 N while sawing |
| Another robot link touching the food | any link of either arm counts as touching: 0 |
| A dropped or flung knife passing through the food | cuts only with the knife in a hand (a gripper within 3 cm of its handle) moving under 0.5 m/s |
| Fork left standing in the rest of the sausage | not laid down: at most 0.75 |

## Real setup and parts

| Part | Source |
| --- | --- |
| Smoked bratwurst links, fully cooked | Johnsonville Smoked Bratwurst, 5 links, 14 oz (about 80 g each), any US grocery |
| Dinner plate 26 cm, glazed white | Corelle Winter Frost White 10-1/4 in dinner plate (or any 26 cm glazed plate with a 19 cm flat floor) |
| Non-slip mat | Dycem non-slip mat (or any silicone non-slip mat) under the plate |
| Serrated table knife 24 cm | Oneida Chef's Table B327KDTF 9-1/2 in dinner knife (serrated, round tip) |
| Dinner fork | the matching Oneida Chef's Table dinner fork (19 cm, four tines) |
| Knife rest | wooden: 11 x 5 x 0.4 cm base, two 11 x 0.8 x 2.4 cm walls 4 mm apart (or any slotted rest that holds the knife edge down) |
| Fork rest | wooden block 5 x 1.6 x 1.15 cm under the fork's handle |

Reset: a fresh link from the pack each trial, trimmed to 13 cm (fridge-cold or room temperature, never heated); wipe
the plate, wash the fork and knife; the grippers never touch the food.

## Licence

All geometry is procedural (boxes and cylinders in `task.py`); no third-party assets.
