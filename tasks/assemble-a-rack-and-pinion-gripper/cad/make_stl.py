"""Write printable STLs (mm) of the housing, both racks and the pinion's disc and hub, from the same boxes and cylinders
task.py simulates (overlapping shells; slicers merge them). The dowel pins are bought, not printed. Run from the task
folder:  python cad/make_stl.py   (writes cad/*.stl). gripper.scad is the editable source with true chamfers."""
from __future__ import annotations

import math
import pathlib
import struct
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import task  # noqa: E402


def box(c, s):
    x0, y0, z0 = (c[i] - s[i] / 2 for i in range(3))
    x1, y1, z1 = (c[i] + s[i] / 2 for i in range(3))
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6),
         (3, 0, 4), (3, 4, 7)]
    return [tuple(v[i] for i in t) for t in f]


def cyl(c, d, h, axis, n=40):
    r = d / 2
    tris = []
    def pt(a, t):
        u, w = r * math.cos(a), r * math.sin(a)
        if axis == "Z":
            return (c[0] + u, c[1] + w, c[2] + t)
        if axis == "Y":
            return (c[0] + u, c[1] + t, c[2] + w)
        return (c[0] + t, c[1] + u, c[2] + w)
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        b0, b1, t0, t1 = pt(a0, -h / 2), pt(a1, -h / 2), pt(a0, h / 2), pt(a1, h / 2)
        cb, ct = pt(0, -h / 2), pt(0, h / 2)
        cb = tuple(cb[k] - (pt(0, -h / 2)[k] - pt(math.pi, -h / 2)[k]) / 2 for k in range(3))
        ct = tuple(ct[k] - (pt(0, h / 2)[k] - pt(math.pi, h / 2)[k]) / 2 for k in range(3))
        tris += [(b0, b1, t1), (b0, t1, t0), (cb, b1, b0), (ct, t0, t1)]
    return tris


def shapes(items, origin=(0, 0, 0)):
    tris = []
    for it in items:
        c, s = it[0], it[1]
        o = it[3] if len(it) > 3 else {}
        if o.get("collide", True) is False:
            continue
        c = tuple(1000 * (c[i] + origin[i]) for i in range(3))
        s = tuple(1000 * v for v in s)
        if o.get("shape") == "cyl":
            ax = o.get("axis", "Z")
            i = "XYZ".index(ax)
            tris += cyl(c, max(v for j, v in enumerate(s) if j != i), s[i], ax)
        elif o.get("shape") == "sphere":
            continue  # pin tips: the dowels' own rounded ends
        else:
            tris += box(c, s)
    return tris


def write(name, tris):
    with open(HERE / f"{name}.stl", "wb") as f:
        f.write(name.encode().ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            f.write(struct.pack("<3f", 0, 0, 0))
            for v in t:
                f.write(struct.pack("<3f", *v))
            f.write(b"\0\0")
    print(name, len(tris), "triangles")


write("housing", shapes(task.HOUSING.boxes))
write("rack_near", shapes(task.NEAR.boxes))
write("rack_far", shapes(task.FAR.boxes))
write("pinion_disc_and_hub", shapes([b for b in task.PINION.boxes if not (b[1][0] == task.PIN_D)]))
