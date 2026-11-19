# Set a short NMR sample's depth, then load the spinner into the changer

Every compound a synthesis lab makes is identified by solution NMR, and the user sets each sample's depth in its spinner
by hand with the facility's depth gauge. A standard 40-50 mm sample is pushed to the gauge's base; a shorter one has to be
centred on the gauge's line instead (Cornell NMR facility, https://nmr.chem.cornell.edu/?p=103 and
https://nmr.chem.cornell.edu/new-users/training/nmr-sample-preparation; Caltech, "Florence" operating guide,
https://cce.caltech.edu/documents/2982/florence-operation.pdf: "if your sample's liquid column is short, move the tube up in
the spinner turbine to keep it centered around the probe center line marked on the depth gauge"). The tube is then left
alone in the spinner ("Once the sample depth has been set, do NOT move the tube inside the spinner", Cornell), and the
spinner is carried by its body to the sample changer.

The robot picks the 5 mm glass tube from a rack by its sleeved top, lowers it into the spinner seated in the gauge,
pushes it through the spinner's gripping collar (a few newtons of friction, stick-slip) and stops with the 30 mm sample
centred on the line within 1 mm: the gauge's floor is 20 mm below the line, so pushing to the floor leaves the column
4 mm low. Then it lifts the spinner straight up by its turbine until the tube clears the gauge, carries it round to the
sample changer's carousel and lowers it between two occupied positions onto the open seat, whose bore has no floor near the tube's
end.

## Run

```
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 -E control_hz=20 \
  --instruction "$(cat instruction.txt)" --scorer task_score --max-steps 2400
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  --instruction "$(cat instruction.txt)" --scorer task_score --max-steps 2400
```

Run limit 2,400 steps (120 s at 20 Hz). One trial takes about 4-6 minutes of wall time on one GPU. No seeds: the layout is
fixed (`object_jitter=0`). `TASK_TRACE=<file>` logs the state every 10 steps. Offline tests of the scorer (no simulator):
`python -m pytest -q test_task.py`.

Demo variants for the simulation checks (environment variables): `TOUCH=low` holds the tube 60 mm above its bottom
(below the red band: scores 0), `TOUCH=floor` pushes the tube to the gauge's floor (0.25), `TOUCH=tube` lifts the
spinner by gripping the tube (0). `PUSH_OFFSET` (m) is the extra push command for the arm's give, `GRIP_AT` where the
tube is held.

## Scoring

See task.json. Real-trial checks: the depth is read on the mm scale in the gauge's front window or on the open carousel
position's own window, from a front photo (the column's top and bottom are both on the scale; the centre is their mean);
"seated" is the shoulder resting flat on the seat (photo); everything else is judged on the trial video: a finger below the
red band, any other robot part on the tube, the spinner rising off a seat without a gripper on its turbine, a neighbour
moving, a drop. The 1 mm band is our threshold: about the reading resolution of a mm scale by eye.

## Real setup and parts

| Part | Source | Approx. cost |
|---|---|---|
| NMR tube, 5 mm x 178 mm (7 in), 0.43 mm wall, with cap | Wilmad-LabGlass 528-PP-7 | USD 8 |
| Spinner turbine for Bruker 5 mm | Wilmad STB-5 (Sigma-Aldrich Z558117), or Bruker Z42516 | USD 207 |
| Two more spinners and tubes for the neighbouring positions | as above | USD 430 (or 3D-printed dummies of the same outline) |
| Depth gauge, sample changer carousel segment (three seat columns), tube rack | 3D-printed (PETG): STL files in `stl/` (regenerate with `python make_parts.py`, which needs the simulation package) | USD 5 |
| Heat-shrink sleeve 6 mm (dark green), red paint marker, Parafilm, food colour | any | USD 10 |

The tube is filled with 0.40 mL of water with orange food colour (a 30 mm column), capped and the cap sealed with
Parafilm, so nothing can spill on the rig; one tube serves every trial. The sleeve covers only the top 50 mm (the grasp
zone) and the red paint-marker ring marks the lower 80 mm, the part that goes into the probe. Gripper force: cap the YAM gripper at
about 15 N on the tube (a 5 mm borosilicate tube held over 10 mm of sleeve takes this easily). If a tube cracks or
breaks, stop the trial and clear the shards before going on. A same-size polymer tube (5 mm FEP/PTFE NMR tube liner of
the same length) is an acceptable stand-in for a rig that must avoid glass.

## Simulation

`nmrkit.py` builds the fixtures: the gauge and the carousel segment are fixed bodies; the spinner is a loose body whose
bore has two rings of four sprung rollers (the collar); the tube and the two neighbours are loose rigid bodies. PhysX
reports which robot links touch the tube and the spinner. See the fidelity field for what is approximated.

## Results

Both on the final files (task.py, nmrkit.py, instruction.txt), 1280 x 704, `object_jitter=0`:

- Scripted demo (`yam_scripted`): **1.0**, "slot, column 0.2 mm off the line"; the episode ends in success at about
  44 s (880 steps). The right arm picks the tube by its sleeve, pushes it through the spinner in the gauge (the push is
  commanded 21.8 mm below the target to allow for the arm's give against the collar's friction), lets go; the left arm
  grips the turbine, lifts the spinner until the tube clears the gauge, carries it round the carousel and lowers it
  between the neighbours onto the open seat.
- Dream-1 (`dream`, 2,400 steps): **0.0** (run before the last scorer change, which only tightened the carried-by-the-tube check; Dream-1 never touched the tube or the spinner, so its score cannot change). The right arm (right of the video) rose over the tube rack and hovered 6-7 cm
  from the tube without gripping it; the left arm reached toward the carousel at about 30 s and drew back; nothing was
  touched.

Offline: `python -m pytest -q test_task.py` (18 tests): the start scores 0; the right end state 1.0 (0.75 in the gauge);
the bypasses score 0 or are capped: pushed to the gauge's floor (0.25), set beside or above the seat, a hand still on
it, a finger below the red band, another robot link on the tube, the spinner carried by the tube, a neighbour knocked,
a dropped tube put back, a tube beside the spinner, and a set depth then undone (the end state counts).

## Third-party assets

None: every part is built from primitive shapes in `task.py`.
