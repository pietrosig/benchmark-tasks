"""Write STL files for the printed parts (depth gauge, changer rack segment, tube rack) from the same geometry the
simulation uses (task.py), so a real setup matches it. Run: python make_parts.py [out_dir]  (default: ./stl)

Print in PETG, 0.2 mm layers, 100 % infill for the feet (they weigh the parts down; tape or clamp the feet to the bench).
The window is left open in print: glue a 3 x 12 mm strip of clear PETG sheet over it, and draw the scale and the black
line with a fine marker against a printed ruler, or print the scale as a sticker. Drawn-only items (scale ticks, the
line) are skipped here."""

from __future__ import annotations

import math
import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import task as T  # noqa: E402


def _box_tris(c, s, rz_deg=0.0):
    hx, hy, hz = (v / 2 for v in s)
    a = math.radians(rz_deg)
    ca, sa = math.cos(a), math.sin(a)
    pts = []
    for dx in (-hx, hx):
        for dy in (-hy, hy):
            for dz in (-hz, hz):
                pts.append((c[0] + ca * dx - sa * dy, c[1] + sa * dx + ca * dy, c[2] + dz))
    # vertex index = 4*ix + 2*iy + iz
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for f in faces:
        tris.append((pts[f[0]], pts[f[1]], pts[f[2]]))
        tris.append((pts[f[0]], pts[f[2]], pts[f[3]]))
    return tris


def write_stl(path, boxes):
    tris = []
    for c, s, _k, o in boxes:
        if not o.get("collide", True) or o.get("shape") == "cyl":
            continue
        tris += _box_tris(c, s, o.get("rz", 0.0))
    with open(path, "wb") as f:
        f.write(b"robocurve task part".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            f.write(struct.pack("<3f", 0, 0, 0))
            for p in t:
                f.write(struct.pack("<3f", *(1000.0 * v for v in p)))  # mm
            f.write(b"\0\0")
    return len(tris)


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "stl")
    out.mkdir(exist_ok=True)
    solid = lambda bs: [b for b in bs if len(b[2]) == 3]  # noqa: E731  (the clear window is a separate sheet)
    for name, boxes in (("depth_gauge", T._G), ("changer_rack", T._C), ("tube_rack", T._R)):
        n = write_stl(out / f"{name}.stl", solid(boxes))
        print(f"{name}.stl: {n} triangles (overlapping boxes; most slicers merge them)")


if __name__ == "__main__":
    main()
