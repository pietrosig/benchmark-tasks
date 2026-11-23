// stands.scad: the 3D-printed parts of the beam-walking bench (OpenSCAD; mm). Print in PLA, 0.2 mm layers,
// 4 perimeters, 40 % infill, each standing on its foot. Pick a part with -D 'part="aperture"' (or laser, photodiode).
//   openscad -D 'part="aperture"' -o aperture.stl stands.scad    (print two)
// Heights are from the top of the 1/2 in (12.7 mm) board; the beam runs 187.3 mm above it (200 mm above the bench).
// Each foot is screwed to the board with two 4 mm wood screws through its slots.

part = "aperture";
BEAM = 187.3;          // beam height above the board
$fn = 64;

module foot() {        // 40 (along the beam) x 50 x 5 mm, two slots for 4 mm wood screws
    difference() {
        translate([-20, -25, 0]) cube([40, 50, 5]);
        for (y = [-19, 19]) hull() for (x = [-8, 8]) translate([x, y, -1]) cylinder(d = 4.5, h = 7);
    }
}

module column(top) { translate([-10, -10, 0]) cube([20, 20, top]); }

// Alignment aperture: a 40 x 40 x 3 mm plate (print in white PLA, so the red spot shows) with its hole centre at the
// beam height. The hole is printed as a 0.6 mm pilot and drilled through with a 0.8 mm bit (a standard PCB drill); a
// Ø6 mm counterbore from the back leaves the plate 1 mm thick at the hole. The plate's front face (toward M2) is at
// x = -10, flush with the column's front.
module aperture() {
    foot();
    column(BEAM - 20);
    difference() {
        translate([-10, -20, BEAM - 20]) cube([3, 40, 40]);
        translate([-11, 0, BEAM]) rotate([0, 90, 0]) cylinder(d = 0.6, h = 5);
        translate([-9, 0, BEAM]) rotate([0, 90, 0]) cylinder(d = 6, h = 5);
    }
}

// Laser holder: a 16 x 20 x 20 mm block on the column, a Ø9.2 mm bore along x at the beam height for the
// Ø9 x 20 mm module, an M3 grub-screw hole from the top to clamp it (tap the plastic or use an M3 heat-set insert).
module laser() {
    foot();
    column(BEAM - 10);
    difference() {
        translate([-8, -10, BEAM - 10]) cube([16, 20, 20]);
        translate([-9, 0, BEAM]) rotate([0, 90, 0]) cylinder(d = 9.2, h = 18);
        translate([0, 0, BEAM]) cylinder(d = 2.5, h = 11);
    }
}

// Photodiode holder (print in black PLA): a 16 x 25 x 25 mm block; a 5.6 x 4.6 mm pocket, 2 mm deep, in its front face
// (toward aperture 2) takes the BPW34 with its 2.65 mm square active area centred on the beam; two Ø1.2 mm holes take
// its leads (5.08 mm apart) out of the back to the multimeter leads. A light shroud, a tube 16 mm outside and 10 mm
// inside, 31 mm long, runs from the block's front to 1 mm behind aperture 2's plate, so room light can't reach the
// photodiode (the holder's front is 32 mm behind the plate's back).
module photodiode() {
    foot();
    column(BEAM - 12.5);
    difference() {
        translate([-39, 0, BEAM]) rotate([0, 90, 0]) cylinder(d = 16, h = 32);
        translate([-40, 0, BEAM]) rotate([0, 90, 0]) cylinder(d = 10, h = 34);
    }
    difference() {
        translate([-8, -12.5, BEAM - 12.5]) cube([16, 25, 25]);
        translate([-9, -2.8, BEAM - 2.3]) cube([3, 5.6, 4.6]);
        for (y = [-2.54, 2.54]) translate([-9, y, BEAM - 3.5]) rotate([0, 90, 0]) cylinder(d = 1.2, h = 20);
    }
}

if (part == "aperture") aperture();
if (part == "laser") laser();
if (part == "photodiode") photodiode();
