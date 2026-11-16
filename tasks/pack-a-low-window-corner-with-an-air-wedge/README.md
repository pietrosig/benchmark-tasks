# Pack a window's low corner level with an air wedge

A white uPVC window, 91.5 x 60 cm and 14 kg, stands in a timber opening on the bench (the bench top is the opening's
sill). Its right corner is already on a 6 mm packer; its left corner is down on the sill, 6 mm low. The robot has to get
that corner up onto the loose packer, which is how every window is levelled before it is fixed. It cannot lift the window:
the frame is too heavy and has nothing to hold. It uses the fitter's tool, an air wedge: a flat bag that goes into the
gap under the frame, a squeeze bulb that lifts the frame a little with each squeeze, and a button that lets it down again.

What makes it hard for a robot: the empty bag is 2 mm thick and the gap is 3 to 4 mm where it goes in; the first
squeeze only takes up the slack, then the corner rises 2 to 3.5 mm per squeeze; the packer only fits once the corner is
more than 6 mm up; the bag can't be pulled out until the button has let the window down; and both hands are needed, one
on the wedge and one on the packer, close to a window that stands over them.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the opening, the window, the packers, the air wedge, `score()`, `success()`, `distance()` and `demo()` |
| `wedgekit.py` | Fixtures with moving parts for yam_arena task files (joints with springs and limits; the bag's pump and release) |
| `test_task.py` | Offline checks: the file loads, every scoring rule, the gaps, the demo's reach and joint limits |

`wedgekit.py` hooks `yam_arena.scene` when it is imported, without modifying yam_arena, so the task runs with the standard
`yam_arena` embodiment and the `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "The window's left corner is sitting low on the sill. Get it level on the loose packer, using the air wedge to lift it, and leave nothing but the packers under the window." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/air-wedge-dream
```

`task_score` is 1.0 when the trial ends in success, otherwise `score()` at the end of the episode.

One Dream-1 (`dreamlabs/dream-1-beta`) trial with this command scored **0** (`outcome: hit step limit`, `task_score: 0`): it
ran the full 2,400 steps without lifting the window. The left arm (on the left of the video) reached toward the bag in
the first 20 s without touching it (8.5 cm from the grip block at its closest), then moved over the loose packer and
nudged it about 1 cm at 47 s. The right arm (on the right of the video) came down over the bulb between about 35 and
50 s (5.5 cm from it at its closest) and never touched it. From about 65 s both arms held still above the bench. The
bag and the bulb never moved, the bulb was never squeezed, and the corner stayed on the sill (0 mm).

- Run limit: 2,400 steps (2 minutes at 20 Hz).
- Wall time per trial on one L40S GPU, including Isaac Sim start-up: about 3 minutes for the scripted demo and 8
  for a Dream-1 trial.
- The layout is fixed with `-E object_jitter=0`, so every trial starts the same. There are no seeds.
- Set `TASK_TRACE=/some/file.jsonl` to log the corner's height, the bag's lift, the squeezes, the button, and where the
  packer, the bag and the grippers are, every 10 scorer calls (`TRACE_EVERY` changes that; for debugging, it doesn't
  change the trial).

## How the scene works

- **The window** is one rigid 14 kg body on a hinge at its packed right corner: it can rock in its own plane, from its
  left corner on the sill (6 mm low) up to the opening's head (10 mm above level). The right packer is fixed.
- **The gap** under the bottom rail tapers from 6 mm at the right packer to nothing at the left corner. The loose packer
  (6 mm) does not fit under the left corner until the corner is more than 6 mm up.
- **The air wedge** is a loose object: a 16 x 15 cm bag 2 mm thick, an orange grip block on its hose socket, a 75 mm hose
  and an 80 x 45 mm bulb. A full squeeze of the bulb (its side pushed in 9 mm or more of its 12 mm, about 20 N; it has to
  be let out again before the next one counts) adds 1.2 mm to the bag's thickness, up to 12 mm. The bag fills, and
  empties while the red button is held in at least 1.5 mm of its 3 mm (3 N), at 8 mm/s. The bag pushes its top sheet up
  like a stiff spring, so it lifts whatever is on it: under the bottom rail it carries about 170 N, and it cannot be
  pulled out while it does.
- **Where the bag goes matters.** In the demo it sits 35 cm from the right packer, where the gap is 3.6 mm. The first
  squeeze takes up the slack; after two, three, four and five squeezes the corner is 1.6, 4.1, 6.5 and 10.0 mm up. Four
  are not enough for the 6 mm packer to slide in freely; five are.
- **The hose** is three joints and two stiff links: it can droop, lift and bend sideways, so the bulb lies on the sill
  under its own weight and a hand holding the bulb does not hold the bag.

## Scoring

`score()` looks at the end state (`SCORE_OVER = "end"`). Corner height is the gap between the sill and the underside of
the frame at its left end: 0 at the start, 6 mm on a packer.

The packer is in place when it lies flat on the sill under the frame's left jamb: at least 4 cm of its length under the
frame, its centre line 1.4 to 6 cm in from the frame's left end, pointing across the frame (within 20 degrees).

| Score | End state |
|---|---|
| 1.0 | The corner rests on the packer (packer in place, corner height 4.5 to 7.5 mm), the whole bag is on the room side of the frame's face, and both grippers are 5 cm or more from the frame, the packer, the bag and its block, the hose and the bulb |
| 0.75 | The corner rests on the packer, but part of the bag is still under the frame or a gripper is within 5 cm |
| 0.5 | The packer is in place but the corner is more than 7.5 mm up: still held by the wedge |
| 0.25 | The corner is 3 mm or more above the sill and none of the above applies: lifted, not packed |
| 0 | Anything else: nothing moved, the corner back down on the sill, a packer against or beside the frame |

`success()` is the 1.0 state held for 1 s, which ends the trial. With the bag out and the grippers away, nothing but the
packer can be holding the corner up, so 1.0 cannot be reached with the window still on the wedge. In a real trial the
corner height is read with a steel rule or a 5 mm and a 7 mm feeler; the packer's position with the rule.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "The window's left corner is sitting low on the sill. Get it level on the loose packer, using the air wedge to lift it, and leave nothing but the packers under the window." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/air-wedge-demo
```

Open loop: fixed waypoints, nothing read back from the simulator.

1. The right hand grips the wedge's orange block and slides the bag 8.5 cm forward, under the bottom rail.
2. It moves back to the bulb and squeezes it five times. The corner comes up to 10.0 mm.
3. The left hand, open and turned so that one finger leads, pushes the loose packer along the sill with that fingertip
   until 4.8 cm of it is under the corner. (A hand pointing down cannot get its wrist closer than 10 cm to the window, so
   it pushes with the finger that is 5 cm ahead.)
4. The right hand presses the release button with closed fingertips and holds it for 2.5 s. The window comes down on the
   packer.
5. The right hand grips the block again and pulls the bag 11 cm back, out from under the frame, lets go and moves 19 cm
   out to the side, so the wedge is left lying free in front of the window.

Result with this command: `outcome: succeeded`, `task_score: 1`, after 824 steps (41.2 s), well within the 2,400-step
limit. After the five squeezes the corner was 0, 1.6, 4.1, 6.5 and 10.0 mm up. The packer went 4.8 cm under the corner.
With the button held the window came down and settled on the packer with the corner 5.85 mm up. The bag came out and
the hands were clear at 40.2 s; the score held 1.0 for 1 s and the trial ended.

## Real setup

Everything is a standard part.

- **Bench:** at least 105 cm wide, centred on the arms, and reaching 50 cm forward of the arm bases, so the opening
  stands on it.
- **Opening:** two 2x4 posts and a 2x4 head screwed together and to the bench, 93.5 cm wide and 61.6 cm high inside.
  Four 30 x 70 mm steel mending plates on the head, two per face, hang 4 cm below it and keep the window from tipping
  out; add a cord from the window's top rail to the head as a second restraint.
- **Window:** any uPVC fixed light close to 915 x 600 mm with a 70 mm frame and a flat underside (about 14 kg with a
  24 mm double-glazed unit). Order it with toughened or laminated glass.
- **Packers:** two plastic flat packers, 100 x 28 x 6 mm (window fitters' frame packers).
- **Air wedge:** a 160 x 150 mm inflatable lifting bag with a squeeze bulb and a push-button release (rated 135 kg,
  2 mm thick empty). Clamp a 25 x 20 x 22 mm block round the hose where it enters the bag, so a parallel gripper has
  something to hold, and lay the bulb with its button up.
- **Marks:** tape marks on the bench for the loose packer (centre x 31, y 42), the bag (centre x 30.5, y -8) and the
  bulb (centre x 7.5, y -8), in cm from the midpoint between the arm bases.
- **Check before each session:** with the right corner on its packer and the left corner on the sill, the gap under the
  bottom rail is 6 mm at the right packer and closes to nothing at the left end.

The arms never carry the window. The wedge lifts one corner by 16 mm at most (the head stops it), and if it lets go the
corner drops that far onto the sill or the packer.

## Offline checks (no GPU)

```bash
git clone https://github.com/kairosiann/yam-isaac-lab-arena ~/yam-isaac-lab-arena
PYTHONPATH=~/yam-isaac-lab-arena uv run --python 3.12 --with numpy --with pytest python -m pytest test_task.py
```
