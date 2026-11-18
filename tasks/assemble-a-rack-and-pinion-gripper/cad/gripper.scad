// Rack-and-pinion parallel gripper for the task (all sizes in mm, matching task.py).
// Export each part:  openscad -o housing.stl -D 'part="housing"' gripper.scad   (also "rack_near", "rack_far", "pinion")
// Print: PETG, 0.2 mm layers, 4 perimeters, 40 % gyroid infill, no supports (racks print lying on their side face).
// Hardware: 8 x dowel pin ISO 8734 / DIN 6325 m6, 3 x 16 mm, hardened (Zoro Select M390800300016); 1 x shoulder screw ISO 7379, 5 mm shoulder x 16 mm, M4 thread (McMaster-Carr 92981A series); 1 x steel flat
// bar 160 x 50 x 6 mm glued under the housing (brings the housing to about 460 g).
part = "housing";

R = 12;            // pinion pitch radius: pins on a 24 mm circle
P = 2 * PI * R / 8; // 9.42 mm circular pitch (module 3)
GAP = 3.4;         // tooth gap round a 3 mm pin: 0.4 mm backlash per rack
TW = P - GAP;      // tooth width
T_IN = R - 1.0;    // tooth crest, 1 mm inside the pitch line
T_OUT = R + 1.8;   // tooth root (rack body face)
BODY_W = 8; RL = 88; RH = 12;
CLOSED_C = 4;      // closed rack's middle, 4 mm from the axis along the rack
HL = 160; HW = 52; FLOOR = 4; WALL_IN = 22.5; WALL_T = 3.5; WALL_H = 12; RAIL_H = 5;
DISC_D = 27.2; DISC_T = 4;
$fn = 48;

module chamfered_box(size, c) {  // a box with its top edges chamfered by c
    hull() {
        translate([0, 0, 0]) cube([size[0], size[1], size[2] - c]);
        translate([c, c, 0]) cube([size[0] - 2 * c, size[1] - 2 * c, size[2]]);
    }
}

module rack(side) {  // side = +1 far (orange), -1 near (white); origin on the pitch line, rack middle, underside
    g0 = -side * CLOSED_C;
    mirror([side < 0 ? 1 : 0, 0, 0]) {
        translate([T_OUT - R, -RL / 2, 0]) cube([BODY_W, RL, RH]);                      // body
        for (k = [-6 : 6]) {
            yc = g0 + (k + 0.5) * P;
            if (abs(yc) <= RL / 2 - TW / 2)
                hull() {                                                                 // tooth with a rounded crest
                    translate([T_IN + 0.6 - R, yc - TW / 2, 1.5]) cube([T_OUT - T_IN - 0.6, TW, RH - 1.5]);
                    translate([T_IN - R, yc - TW / 2 + 0.6, 1.5]) cube([0.6, TW - 1.2, RH - 1.5]);
                    translate([T_IN - R, yc - TW / 2 + 0.8, 0]) cube([T_OUT - T_IN, TW - 1.6, 1.5]); // lead-in foot
                }
        }
        translate([T_OUT - R + BODY_W / 2 - 3.75, RL / 2, 0]) cube([7.5, 6, 30]);           // finger plate
        difference() {                                                                   // finger-mount boss
            translate([T_OUT - R + BODY_W / 2 - 3.5, -3, RH]) cube([7, 6, 19.5]);
            translate([T_OUT - R + BODY_W / 2, -4, RH + 12]) rotate([-90, 0, 0]) cylinder(d = 2.5, h = 8); // M3 tap hole
        }
    }
}

module housing() {
    difference() {
        union() {
            translate([-HW / 2, -HL / 2, 0]) cube([HW, HL, FLOOR]);
            for (s = [-1, 1]) {
                translate([s > 0 ? WALL_IN : -WALL_IN - WALL_T, -HL / 2 + 5, 0]) chamfered_box([WALL_T, HL - 10, FLOOR + WALL_H], 1.2);
                translate([s > 0 ? T_OUT + 0.2 : -WALL_IN, -HL / 2 + 5, 0]) cube([WALL_IN - T_OUT - 0.2, HL - 10, FLOOR + RAIL_H]);
            }
            translate([-WALL_IN, RL / 2 - CLOSED_C, 0]) cube([WALL_IN - T_OUT - 0.4, 4, FLOOR + RAIL_H + 9]);       // near stop
            translate([T_OUT + 0.4, -(RL / 2 - CLOSED_C) - 4, 0]) cube([WALL_IN - T_OUT - 0.4, 4, FLOOR + RAIL_H + 9]); // far stop
        }
        translate([0, 0, -1]) cylinder(d = 4.2, h = FLOOR + 2);   // M4 thread clearance (tapped into the steel bar below)
    }
}

module pinion() {
    difference() {
        union() {
            cylinder(d = DISC_D, h = DISC_T);
            cylinder(d = 10, h = 14);
        }
        translate([0, 0, -1]) cylinder(d = 5.1, h = 16);                                // shoulder screw bore
        for (k = [0 : 7]) rotate([0, 0, k * 45]) translate([R, 0, 0.5]) cylinder(d = 2.95, h = DISC_T); // press fit, 3.5 mm deep
    }
}

if (part == "housing") housing();
if (part == "rack_near") rack(-1);
if (part == "rack_far") rack(1);
if (part == "pinion") pinion();
