"""Geometry of the dental training head and the bitewing holder (pure NumPy, no simulator imports).

Bench coordinates (metres): x forward from the midpoint between the arm bases, y toward the left arm, z above the bench
top. The head faces the robot (its front is -x), upright, the biting plane level. The patient's right is +y (the robot's
left). The lower jaw turns about the hinge axis (parallel to y) at HINGE; jaw angle 0 is closed (teeth together),
negative is open.

The head is an ANNWAH dental training head (typodont of 28 resin teeth in aluminium jaws on a hinged articulator, a
silicone face mask), built from its published size and a standard adult typodont's tooth sizes (README). Tooth crowns
are rounded convex blocks at true size; the face is one smooth signed-distance shape drawn as a mesh.
"""
from __future__ import annotations

import math

import numpy as np

# -- placement --------------------------------------------------------------------------------------------------------
XI = 0.260  # front (labial face) of the lower central incisors, jaw closed, in the head frame
ZO = float(__import__("os").environ.get("HEAD_ZO", 0.130))  # the lower biting (occlusal) plane, jaw closed
HINGE0 = np.array([XI + 0.092, 0.0, ZO + 0.030])  # the articulator's hinge axis (y), head frame
REC = math.radians(float(__import__("os").environ.get("HEAD_REC", 25.0)))  # the head is reclined this far (chin up)
PIV = np.array([XI + 0.115, 0.0, ZO - 0.100])  # the mount's ball joint, under the back of the neck
RREC = np.array([[math.cos(REC), 0, math.sin(REC)], [0, 1, 0], [-math.sin(REC), 0, math.cos(REC)]])


def to_world(p):
    """Head frame (upright, as the dimensions below are given) to bench coordinates (reclined on its mount)."""
    return PIV + (np.asarray(p, float) - PIV) @ RREC.T


def from_world(p):
    return PIV + (np.asarray(p, float) - PIV) @ RREC


HINGE = to_world(HINGE0)  # the hinge axis in bench coordinates
OPEN = math.radians(30.0)  # the jaw starts open this far: 48 mm between the incisal edges
JAW_FRICTION = (0.45, 0.38)  # N m static, moving: the hinge's friction screw, set so the jaw stays where it is put
JAW_MASS = 0.15  # kg: lower jaw plate, typodont teeth and the mask's chin

TEETH_C, GUM_C, SKIN_C, INNER_C, METAL_C = ((0.98, 0.97, 0.93), (0.86, 0.50, 0.53), (0.89, 0.72, 0.62),
                                            (0.62, 0.30, 0.32), (0.66, 0.67, 0.69))

# -- typodont: tooth centres (back from XI, toward the patient's right), mesiodistal, buccolingual, crown height (mm) ----
LOWER = [  # central incisor .. second molar, patient's right side; mirrored for the left
    ("41", 2.6, 2.7, 5.0, 6.0, 9.0), ("42", 6.0, 8.0, 5.5, 6.2, 9.0), ("43", 10.8, 12.6, 7.0, 7.6, 10.0),
    ("44", 16.8, 16.4, 7.0, 7.6, 8.0), ("45", 23.6, 18.8, 7.1, 8.0, 7.6), ("46", 32.6, 21.4, 11.0, 10.4, 7.2),
    ("47", 43.4, 23.0, 10.5, 10.0, 7.0)]
UPPER = [
    ("11", 0.6, 4.3, 8.5, 7.0, 10.0), ("12", 5.4, 11.4, 6.5, 6.2, 9.0), ("13", 11.0, 16.4, 7.5, 8.0, 10.0),
    ("14", 17.2, 19.8, 7.0, 9.0, 8.0), ("15", 24.0, 21.8, 6.5, 9.0, 7.6), ("16", 32.4, 24.4, 10.0, 11.0, 7.0),
    ("17", 42.4, 25.6, 9.0, 10.6, 6.8)]

# -- the bitewing holder (Rinn XCP-ORA arm and ring, XCP bitewing bite block, size 2 phosphor plate), in its own frame:
# u along the plate's long edge toward the front of the mouth, v up, n out of the plate's face toward the teeth (the
# beam axis). Origin: the plate's centre. mm.
PLATE = (41.0, 31.0, 2.0)  # size 2 phosphor plate in its barrier envelope
HOLDER_BOXES = [  # (name, u range, v range, n range, colour)
    ("plate", (-20.5, 20.5), (-15.5, 15.5), (-1.0, 1.0), (0.30, 0.42, 0.62)),
    ("slot", (4.0, 20.0), (-1.5, 6.0), (-2.6, -1.0), (0.80, 0.14, 0.14)),  # the bite block's slot round the plate
    ("tab", (4.0, 20.0), (-1.5, 1.5), (1.0, 17.5), (0.80, 0.14, 0.14)),  # the biting tab
    ("arm_a", (16.0, 73.0), (-3.0, 3.0), (17.0, 21.0), (0.20, 0.38, 0.70)),  # out of the mouth beside the teeth
    ("arm_b", (66.5, 73.5), (-3.5, 3.5), (17.0, 124.0), (0.20, 0.38, 0.70)),  # parallel to the beam, past the ring
    ("ring_slide", (35.0, 67.0), (-3.0, 3.0), (72.0, 78.0), (0.20, 0.38, 0.70)),  # the ring's slide on the arm
]
RING = (31.0, 36.0, 71.0, 79.0)  # inner and outer radius, n range: the aiming ring, centred on the beam axis
ROD_END = np.array([70.0, 0.0, 124.0])  # the arm's straight leg runs along the beam axis out past the ring to here
GRASP = np.array([70.0, 0.0, 105.0])  # the holding hand pinches the arm's leg here, 26 mm outside the ring (u, v, n), mm
HOLDER_MASS = 0.016  # plate in envelope 3 g, bite block 2 g, arm 6 g, ring 5 g
HOLDER_COM = np.array([20.0, 0.5, 42.0])  # its centre of mass (u, v, n), mm, from those parts' masses
TAB_T = 3.0  # biting tab thickness (mm); the jaw closed on it stops this far short of closed at the premolars

# the target: a right premolar bitewing. ALPHA: the premolar row's direction (back and outward) from straight back.
ALPHA = math.radians(18.0)
U_F = np.array([-math.cos(ALPHA), -math.sin(ALPHA), 0.0])  # the plate's long axis, toward the front
V_F = np.array([0.0, 0.0, 1.0])
N_F = np.cross(U_F, V_F) * -1.0  # buccal (toward the cheek): (-sin a, cos a, 0)
D_TARGET = 0.007  # plate centre plane this far lingual of the premolars' lingual surfaces


def _crown(L, B, H, n_u=18, n_v=10, e=0.35):
    """A rounded block (superellipsoid), L along x, B along y, H along z, centred at 0."""
    us = np.linspace(-math.pi, math.pi, n_u, endpoint=False)
    vs = np.linspace(-math.pi / 2, math.pi / 2, n_v)
    sp = lambda w, p: np.sign(w) * np.abs(w) ** p  # noqa: E731
    pts = []
    for v in vs:
        for u in us:
            pts.append([L / 2 * sp(math.cos(v), e) * sp(math.cos(u), e), B / 2 * sp(math.cos(v), e) * sp(math.sin(u), e),
                        H / 2 * sp(math.sin(v), e)])
    pts = np.array(pts)
    faces = []
    for j in range(n_v - 1):
        for i in range(n_u):
            a, b = j * n_u + i, j * n_u + (i + 1) % n_u
            c, d = a + n_u, b + n_u
            faces += [[a, b, d], [a, d, c]]
    return _dedupe(pts, np.array(faces))


def _dedupe(pts, faces):
    key = np.round(pts, 9)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    f = inv.reshape(-1)[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    return uniq, f


def _box_mesh(lo, hi, R=np.eye(3), t=np.zeros(3)):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    f = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3],
                  [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]])
    return c @ np.asarray(R).T + t, f


def _hull_mesh(pts):
    """A closed triangle mesh of the convex hull of a small point set (gift wrapping, O(n^4) fine for < 30 points)."""
    pts = np.asarray(pts, float)
    n = len(pts)
    faces = []
    cen = pts.mean(axis=0)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                nrm = np.cross(pts[j] - pts[i], pts[k] - pts[i])
                if np.linalg.norm(nrm) < 1e-12:
                    continue
                s = (pts - pts[i]) @ nrm
                if np.all(s <= 1e-10):
                    f = [i, j, k]
                elif np.all(s >= -1e-10):
                    f = [i, k, j]
                else:
                    continue
                faces.append(f)
    return pts, np.array(faces)


def _quat_to_R(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _ry(a):
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def _arch(table):
    """Each tooth (both sides): name, centre (x, y) in bench coordinates, tangent angle, size."""
    out = []
    for side in (1, -1):
        pts = [(XI + t[1] / 1000, side * t[2] / 1000) for t in table]
        for i, t in enumerate(table):
            a = pts[max(i - 1, 0)]
            b = pts[min(i + 1, len(pts) - 1)]
            ang = math.atan2(b[1] - a[1], b[0] - a[0])  # direction going back along the arch
            left = {"4": "3", "1": "2"}[t[0][0]]
            name = t[0] if side > 0 else left + t[0][1]
            out.append((name, np.array(pts[i]), ang, t[3] / 1000, t[4] / 1000, t[5] / 1000, side))
    return out


def lower_teeth():
    return _arch(LOWER)


def upper_teeth():
    return _arch(UPPER)


def _mesh_item(pts, faces, color, collide=True, visible=True):
    return ((0.0, 0.0, 0.0), (0.01, 0.01, 0.01), color, {"shape": "mesh", "points": np.asarray(pts, float),
                                                        "faces": np.asarray(faces, int), "collide": collide,
                                                        "visible": visible})


def head_items():
    """Shapes on the fixed head (bench coordinates): upper teeth, upper gum and palate, cheeks, upper lip, face."""
    items = []
    for name, c, ang, L, B, H, side in upper_teeth():
        p, f = _crown(L, B, H)
        R = _rz(ang)
        items.append(_mesh_item(p @ R.T + np.array([c[0], c[1], ZO + H / 2 - (0.002 if name[1] in "12" else 0.0)]), f,
                                TEETH_C))
        lo, hi = np.array([-L / 2 - 0.0002, -B / 2 - 0.001, 0.0]), np.array([L / 2 + 0.0002, B / 2 + 0.001, 0.0])
        lo[2], hi[2] = ZO + H - 0.0015, ZO + 0.030
        p, f = _box_mesh(lo - np.array([0, 0, 0]), hi, R, np.array([c[0], c[1], 0.0]))
        items.append(_mesh_item(p, f, GUM_C))
    # palate: a slab above the upper arch, from the gum line inward
    ring = [(c[0], c[1] - side * (B / 2 + 0.0005) * math.cos(ang), ang, side) for name, c, ang, L, B, H, side in upper_teeth()]
    pal = [[x, y, ZO + 0.020] for x, y, _, _ in ring] + [[x, y, ZO + 0.032] for x, y, _, _ in ring]
    pal += [[XI + 0.050, 0.0, ZO + 0.020], [XI + 0.050, 0.0, ZO + 0.032]]
    p, f = _hull_mesh(pal)
    items.append(_mesh_item(p, f, GUM_C))
    # cheeks: inner walls 7 mm outside the upper molars' buccal faces, from the canines back
    for side in (1, -1):
        y0 = side * (0.016 + 0.004 + 0.007)
        y1 = side * (0.0256 + 0.0053 + 0.007)
        lo_hi = []
        for x, y in ((XI + 0.010, y0), (XI + 0.050, y1)):
            for dy in (0.0, side * 0.006):
                for z in (ZO - 0.018, ZO + 0.026):
                    lo_hi.append([x, y + dy, z])
        p, f = _hull_mesh(lo_hi)
        items.append(_mesh_item(p, f, INNER_C))
    # back of the mouth (throat), dark
    p, f = _box_mesh((XI + 0.090, -0.030, ZO - 0.030), (XI + 0.100, 0.030, ZO + 0.032))
    items.append(_mesh_item(p, f, (0.30, 0.10, 0.12)))
    # upper lip: a collider across the top of the mouth opening (the face mask's lip, 10 mm thick)
    p, f = _box_mesh((XI - 0.014, -0.032, ZO + 0.013), (XI - 0.004, 0.032, ZO + 0.030))
    items.append(_mesh_item(p, f, SKIN_C, visible=False))
    items += face_items(upper=True)
    return [(c_, s_, k_, {**opt, "points": to_world(opt["points"])}) for c_, s_, k_, opt in items]


def jaw_items():
    """Shapes on the lower jaw, in the jaw's own frame (bench coordinates minus HINGE, jaw closed)."""
    items = []
    o = HINGE0
    for name, c, ang, L, B, H, side in lower_teeth():
        p, f = _crown(L, B, H)
        R = _rz(ang)
        items.append(_mesh_item(p @ R.T + np.array([c[0], c[1], ZO - H / 2]) - o, f, TEETH_C))
        lo, hi = np.array([-L / 2 - 0.0002, -B / 2 - 0.001, ZO - 0.026]), np.array([L / 2 + 0.0002, B / 2 + 0.001, ZO - H + 0.0015])
        p, f = _box_mesh(lo, hi, R, np.array([c[0], c[1], 0.0]))
        items.append(_mesh_item(p - o, f, GUM_C))
    # floor of the mouth inside the lower arch (no tongue on this head)
    ring = [(c[0], c[1] - side * (B / 2 + 0.0005) * math.cos(ang)) for name, c, ang, L, B, H, side in lower_teeth()]
    fl = [[x, y, ZO - 0.022] for x, y in ring] + [[x, y, ZO - 0.030] for x, y in ring]
    p, f = _hull_mesh(fl)
    items.append(_mesh_item(p - o, f, GUM_C))
    # lower lip collider (the mask's lip, moving with the jaw)
    p, f = _box_mesh((XI - 0.014, -0.032, ZO - 0.030), (XI - 0.004, 0.032, ZO - 0.010))
    items.append(_mesh_item(p - o, f, SKIN_C, visible=False))
    # chin collider: the mandible's front, for a hand to lift it by
    p, f = _box_mesh((XI - 0.012, -0.028, ZO - 0.062), (XI + 0.030, 0.028, ZO - 0.030))
    items.append(_mesh_item(p - o, f, SKIN_C, visible=False))
    items += [(c_, s_, k_, {**opt, "points": opt["points"] - o}) for c_, s_, k_, opt in face_items(upper=False)]
    return [(c_, s_, k_, {**opt, "points": opt["points"] @ RREC.T}) for c_, s_, k_, opt in items]


# -- the face mask: a smooth signed-distance shape, drawn by rays from inside -------------------------------------------
def _ell(p, c, r):
    q = (p - np.asarray(c)) / np.asarray(r)
    k0 = np.linalg.norm(q, axis=-1)
    k1 = np.linalg.norm(q / np.asarray(r), axis=-1)
    return np.where(k1 > 1e-12, k0 * (k0 - 1.0) / np.maximum(k1, 1e-12), -min(r))


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def _box_sdf(p, lo, hi):
    c, h = (np.asarray(lo) + hi) / 2, (np.asarray(hi) - lo) / 2
    q = np.abs(p - c) - h
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)


MOUTH_W = 0.026  # half-width of the mask's mouth opening (52 mm)


def _head_sdf(p):
    d = _ell(p, (XI + 0.080, 0.0, ZO + 0.070), (0.096, 0.078, 0.096))  # cranium
    d = _smin(d, _ell(p, (XI + 0.030, 0.0, ZO + 0.030), (0.040, 0.056, 0.050)), 0.02)  # mid-face
    for s in (-1, 1):
        d = _smin(d, _ell(p, (XI + 0.030, s * 0.044, ZO + 0.032), (0.030, 0.024, 0.030)), 0.012)  # cheeks
        sock = _ell(p, (XI - 0.004, s * 0.032, ZO + 0.090), (0.012, 0.016, 0.010))
        d = np.maximum(d, -sock + 0.0)  # shallow eye hollows
    d = _smin(d, _ell(p, (XI - 0.016, 0.0, ZO + 0.054), (0.015, 0.013, 0.028)), 0.008)  # nose
    d = _smin(d, _ell(p, (XI - 0.026, 0.0, ZO + 0.040), (0.006, 0.009, 0.006)), 0.006)  # nose tip
    d = _smin(d, _ell(p, (XI + 0.000, 0.0, ZO + 0.100), (0.016, 0.050, 0.010)), 0.012)  # brow ridge
    for s in (-1, 1):
        d = _smin(d, _ell(p, (XI + 0.088, s * 0.077, ZO + 0.058), (0.016, 0.010, 0.032)), 0.005)  # ears
    d = _smin(d, _ell(p, (XI - 0.012, 0.0, ZO + 0.018), (0.012, 0.034, 0.010)), 0.008)  # upper lip
    d = _smin(d, _ell(p, (XI + 0.100, 0.0, ZO - 0.040), (0.042, 0.045, 0.065)), 0.02)  # neck
    return d


def _jaw_sdf(p):
    d = _ell(p, (XI + 0.030, 0.0, ZO - 0.022), (0.042, 0.052, 0.036))
    d = _smin(d, _ell(p, (XI + 0.006, 0.0, ZO - 0.040), (0.020, 0.026, 0.022)), 0.012)  # chin
    d = _smin(d, _ell(p, (XI - 0.010, 0.0, ZO - 0.016), (0.011, 0.032, 0.009)), 0.008)  # lower lip
    return d


def _mouth_region(p):
    """Inside the mouth opening and the cavity behind it (negative inside): the mask has a 52 mm wide opening."""
    return _box_sdf(p, (XI - 0.040, -MOUTH_W, ZO - 0.010), (XI + 0.064, MOUTH_W, ZO + 0.013))


def upper_face_sdf(p):
    d = _head_sdf(p)
    # the part below the lip line in front of the hinge belongs to the jaw
    below = _box_sdf(p, (XI - 0.060, -0.080, ZO - 0.120), (HINGE0[0] - 0.012, 0.080, ZO + 0.006))
    d = np.maximum(d, -below)
    cav = _box_sdf(p, (XI - 0.060, -MOUTH_W, ZO - 0.120), (XI + 0.062, MOUTH_W, ZO + 0.013))
    d = np.maximum(d, -cav)  # the mouth opening up to the upper lip, and the cavity behind it
    neck_cut = p[..., 2] - 0.0  # nothing below the bench
    return np.maximum(d, -neck_cut)


def lower_face_sdf(p):
    d = _jaw_sdf(p)
    keep = _box_sdf(p, (XI - 0.060, -0.080, ZO - 0.120), (HINGE0[0] - 0.012, 0.080, ZO + 0.006))
    d = np.maximum(d, keep)
    cav = _box_sdf(p, (XI - 0.060, -MOUTH_W, ZO - 0.010), (XI + 0.062, MOUTH_W, ZO + 0.020))
    return np.maximum(d, -cav)


def _ray_mesh(sdf, centre, n_u=96, n_v=64, r_max=0.25):
    centre = np.asarray(centre, float)
    us = np.linspace(-math.pi, math.pi, n_u, endpoint=False)
    vs = np.linspace(-math.pi / 2 + 1e-3, math.pi / 2 - 1e-3, n_v)
    U, V = np.meshgrid(us, vs)
    dirs = np.stack([np.cos(V) * np.cos(U), np.cos(V) * np.sin(U), np.sin(V)], axis=-1).reshape(-1, 3)
    t = np.zeros(len(dirs))
    for _ in range(300):
        s = sdf(centre + dirs * t[:, None])
        t = np.where(s < 0, t + np.maximum(-s, 2e-4), t)  # march out from inside to the first exit
        if np.all(s >= -1e-5):
            break
    # refine: bisect between the last inside and the first outside point
    lo = np.maximum(t - 0.004, 0.0)
    hi = t + 0.0
    for _ in range(25):
        mid = (lo + hi) / 2
        inside = sdf(centre + dirs * mid[:, None]) < 0
        lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
    pts = centre + dirs * hi[:, None]
    faces = []
    for j in range(n_v - 1):
        for i in range(n_u):
            a, b = j * n_u + i, j * n_u + (i + 1) % n_u
            c, d = a + n_u, b + n_u
            faces += [[a, d, b], [a, c, d]]
    return pts, np.array(faces)


def _split_inner(pts, faces):
    """Faces on the mouth's cut surfaces (inside the opening) are coloured as the inside of the mouth."""
    cen = pts[faces].mean(axis=1)
    inner = ((np.abs(cen[:, 1]) < MOUTH_W + 0.0005) & (cen[:, 0] > XI - 0.0045) & (cen[:, 0] < XI + 0.063)
             & (cen[:, 2] > ZO - 0.035) & (cen[:, 2] < ZO + 0.025))
    return faces[~inner], faces[inner]


_FACE_CACHE: dict = {}


def face_items(upper: bool):
    key = "u" if upper else "l"
    if key not in _FACE_CACHE:
        if upper:
            pts, faces = _ray_mesh(upper_face_sdf, (XI + 0.075, 0.0, ZO + 0.070))
        else:
            pts, faces = _ray_mesh(lower_face_sdf, (XI + 0.030, 0.0, ZO - 0.035), n_u=80, n_v=48)
        skin, inner = _split_inner(pts, faces)
        _FACE_CACHE[key] = (pts, skin, inner)
    pts, skin, inner = _FACE_CACHE[key]
    out = [_mesh_item(pts, skin, SKIN_C, collide=False)]
    if len(inner):
        out.append(_mesh_item(pts, inner, INNER_C, collide=False))
    return out


# -- jaw kinematics and the target ------------------------------------------------------------------------------------
def jaw_rot(q):
    """The lower jaw's rotation for jaw angle q (about +y; opening is negative)."""
    return _ry(q)


def jaw_to_bench(q, p):
    """A point on the lower jaw, given in the head frame with the jaw closed, in bench coordinates at jaw angle q."""
    return HINGE + jaw_rot(q) @ (RREC @ (np.asarray(p, float) - HINGE0))


def premolar_lingual_line():
    """A point on, and the direction of, the line along the lower right premolars' lingual faces (jaw closed)."""
    t = {name: (c, ang, B) for name, c, ang, L, B, H, side in lower_teeth()}
    pts = []
    for nm in ("44", "45"):
        c, ang, B = t[nm]
        pts.append(np.array([c[0], c[1], ZO]) + np.array([math.sin(ang), -math.cos(ang), 0.0]) * B / 2)
    return pts[0], (pts[1] - pts[0]) / np.linalg.norm(pts[1] - pts[0])


def target_frame():
    """The plate's target pose in the lower jaw's frame (bench coordinates with the jaw closed): centre, [u v n].
    The tab rests on the lower premolars (tab bottom on the biting plane), the plate stands 7 mm lingual of their
    lingual faces and parallel to them, its front edge level with the distal half of the lower right canine."""
    p0, d = premolar_lingual_line()
    canine = {name: c for name, c, ang, L, B, H, side in lower_teeth()}["43"]
    lingual = np.array([d[1], -d[0], 0.0])  # toward the midline
    front = canine[0] + 0.0017
    edge = p0 + (front - p0[0]) / d[0] * d
    centre = edge + d * (PLATE[0] / 2000) + lingual * D_TARGET
    centre[2] = ZO + TAB_T / 2000
    u, v, n = -d, np.array([0.0, 0.0, 1.0]), -lingual
    return to_world(centre), RREC @ np.c_[u, v, n]  # bench coordinates, jaw closed


def holder_items_and_start(tray_centre, start_R):
    """The holder's shapes in its body frame (world-aligned at the start) and the body's start position. start_R is
    the holder frame [u v n] at the start; the body origin is the plate's centre."""
    items = []
    for name, ur, vr, nr, col in HOLDER_BOXES:
        lo = np.array([ur[0], vr[0], nr[0]]) / 1000
        hi = np.array([ur[1], vr[1], nr[1]]) / 1000
        p, f = _box_mesh(lo, hi, start_R)
        items.append(_mesh_item(p, f, col))
    r0, r1, n0, n1 = (v / 1000 for v in RING)
    k = 16
    for i in range(k):
        a0, a1 = 2 * math.pi * i / k, 2 * math.pi * (i + 1) / k
        pts = []
        for a in (a0, a1):
            for r in (r0, r1):
                for nn in (n0, n1):
                    pts.append([r * math.cos(a), r * math.sin(a), nn])
        p, f = _hull_mesh(np.array(pts))
        items.append(_mesh_item(p @ start_R.T, f, (0.20, 0.38, 0.70)))
    return items
