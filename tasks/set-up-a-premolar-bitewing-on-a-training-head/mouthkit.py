"""Fixture kit (one per task folder): a body fixed to the table with moving parts on prismatic or revolute joints (in series if wanted).

A part can latch (``latch_at``), and a joint can be held by narrow limits until named parts have latched (``free_when``,
``free_limits``): a door carried on a six-joint chain is held on its hinges this way and comes free when both release
levers have been pulled. Each state also reports where the two grippers' tool points are and how far their fingers are open.

A part (``Slider``) can have a spring towards ``target``, damping, no gravity, a break point (pulled ``breaks_at`` from rest:
its spring goes limp for good), a slam rule (moving faster than ``slam_speed`` within ``slam_zone`` of rest), or follow
another part (its drive target is ``ratio`` x the leader's position, or ``when_broken`` once the named part breaks).
Box specs: (centre, size, colour[, opts]); opts: shape "box"/"cyl"/"mesh", axis "X"/"Y"/"Z" for cylinders (size = diameter,
diameter, length along the axis, in that axis' order), collide True/False. A 4-value colour is see-through.
A "mesh" shape takes opts "points" (N x 3, in the part's frame, offset by centre) and "faces" (M x 3 triangles); it is
drawn with smooth normals and, if it collides, collides as the convex hull of its points (PhysX convexHull);
"visible": False makes it a collider only. A "light" shape is a sphere light at centre (opts radius, intensity).
A free fixture (fixed=False) may start turned: ``quat`` (x, y, z, w).
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena.task_file import Box, ObjectState


@dataclass(frozen=True)
class Slider:
    name: str
    origin: tuple
    axis: str
    boxes: tuple
    low: float
    high: float
    parent: str | None = None
    kind: str = "prismatic"  # or "revolute" (low/high in rad)
    stiffness: float = 0.0
    damping: float = 1.0
    friction: float = 0.0  # static joint friction effort (N or N m)
    dynamic_friction: float | None = None  # while moving (N or N m); None: the USD default
    mass: float = 0.1
    gravity: bool = True
    target: float = 0.0
    breaks_at: float | None = None
    broken_low: float | None = None
    limp_damping: float = 0.5
    slam_zone: float | None = None
    slam_speed: float | None = None
    follows: str | tuple | None = None
    ratio: float = 1.0
    when_broken: tuple | None = None  # (part name, target once it has broken)
    broken_stiffness: float | None = None  # with when_broken: this joint's spring stiffness from then on
    lock_until: tuple | None = None  # (part name, value): held at 0 until that part's position reaches value
    rate_gain: float = 0.0  # a follower's target also gets rate_gain x the leader's speed
    lock_at: float = 0.0  # where lock_until holds it (default 0)
    retarget: tuple | None = None  # (part, value, new target): once that part reaches value (>= if value >= 0 else <=), target becomes new target (a toggle clicking over)
    relimit: tuple | None = None  # (part name, threshold, low, high): once |that part| >= threshold, new limits (once)
    drive: tuple | None = None  # (speed, part, value): a motor; the target moves at speed (per s) until that part reaches value, then holds (a brake)
    latch_at: float | None = None  # once this part reaches the value (>= if value >= 0 else <=) it counts as latched, for good
    free_when: tuple = ()  # names of parts: once all of them have latched, this joint's limits become free_limits (once)
    free_limits: tuple | None = None  # (low, high) from then on


@dataclass(frozen=True)
class Fixture(Box):
    boxes: tuple = ()
    sliders: tuple = ()
    friction: float = 0.8
    fixed: bool = True  # False: a free object (no anchor to the world)
    base_mass: float = 5.0
    quat: tuple = (0.0, 0.0, 0.0, 1.0)  # starting orientation (x, y, z, w) of a free fixture
    com: tuple | None = None  # the base body's centre of mass in its own frame (default: from its shapes' volumes)


@dataclass(frozen=True)
class FixtureState(ObjectState):
    q: dict = field(default_factory=dict)
    speed: dict = field(default_factory=dict)
    broken: dict = field(default_factory=dict)
    bodies: dict = field(default_factory=dict)  # part name ("body" for the base) -> world position
    latched: dict = field(default_factory=dict)  # part name -> has it reached its latch_at
    hands: dict = field(default_factory=dict)  # "left"/"right" -> {"tcp": xyz between the fingertips, "open": 0 closed .. 1 open}


def fixture(name, pos, boxes, sliders, fixed=True, base_mass=5.0, friction=0.8, quat=(0.0, 0.0, 0.0, 1.0), com=None):
    return Fixture(name=name, pos=tuple(float(v) for v in pos), size=(0.1, 0.1, 0.1), color=(0.7, 0.7, 0.7), boxes=tuple(boxes),
                   sliders=tuple(sliders), fixed=fixed, base_mass=base_mass, friction=friction,
                   quat=tuple(float(v) for v in quat), com=None if com is None else tuple(float(v) for v in com))


_SPECS: dict = {}
_UNLOCKED: dict = {}
_BROKEN: dict = {}
_STARTED: set = set()
_PREV: dict = {}


def _axis_rot(a, t):
    a = np.asarray(a, float); c, s_ = math.cos(t), math.sin(t)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s_ * K + (1 - c) * (K @ K)


def _quat(Rm):
    """Rotation matrix -> (w, x, y, z)."""
    w = math.sqrt(max(0.0, 1 + Rm[0, 0] + Rm[1, 1] + Rm[2, 2])) / 2
    if w > 1e-6:
        return (w, (Rm[2, 1] - Rm[1, 2]) / (4 * w), (Rm[0, 2] - Rm[2, 0]) / (4 * w), (Rm[1, 0] - Rm[0, 1]) / (4 * w))
    i = int(np.argmax(np.diag(Rm))); j, k = (i + 1) % 3, (i + 2) % 3
    r = math.sqrt(max(0.0, 1 + Rm[i, i] - Rm[j, j] - Rm[k, k])); v = [0.0, 0.0, 0.0]
    v[i] = r / 2; v[j] = (Rm[j, i] + Rm[i, j]) / (2 * r); v[k] = (Rm[k, i] + Rm[i, k]) / (2 * r)
    return ((Rm[k, j] - Rm[j, k]) / (2 * r), *v)


def _j(s):
    return f"{s.name}_joint"


def _spawn(prim_path, cfg, translation=None, orientation=None, **kw):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
    from yam_arena.station import linear_rgb

    spec = _SPECS[cfg.fx_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0, 0, 0))))
    qx, qy, qz, qw = orientation if orientation is not None else (0, 0, 0, 1)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(qw, qx, qy, qz))
    mat = f"{prim_path}/Looks/physics"
    m = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    m.func(mat, m)
    looks = {}

    def look(c):
        c = tuple(c)
        if c not in looks:
            p = f"{prim_path}/Looks/c{len(looks)}"
            glass = len(c) == 4
            s = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(c[:3]), roughness=0.05 if glass else 0.4,
                                            metallic=0.0, opacity=c[3] if glass else 1.0)
            s.func(p, s)
            looks[c] = p
        return looks[c]

    def body(path, pos, mass, gravity=True, quat=(1.0, 0.0, 0.0, 0.0)):
        x = UsdGeom.Xform.Define(stage, path)
        x.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
        x.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*quat))
        p = x.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(p)
        UsdPhysics.MassAPI.Apply(p).CreateMassAttr(mass)
        b = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
        b.CreateMaxDepenetrationVelocityAttr(1.0)
        b.CreateSolverPositionIterationCountAttr(32)
        if not gravity:
            b.CreateDisableGravityAttr(True)

    def shapes(path, items):
        for i, item in enumerate(items):
            c, s, k = item[:3]
            o = item[3] if len(item) > 3 else {}
            if o.get("shape") == "light":  # a small round lamp (no collider), e.g. an operatory light
                from pxr import UsdLux
                lt = UsdLux.SphereLight.Define(stage, f"{path}/light_{i}")
                lt.CreateRadiusAttr(float(o.get("radius", 0.03)))
                lt.CreateIntensityAttr(float(o.get("intensity", 30000.0)))
                lt.CreateColorAttr(Gf.Vec3f(*map(float, k[:3])))
                lt.AddTranslateOp().Set(Gf.Vec3d(*c))
                continue
            if o.get("shape") == "mesh":
                pts = np.asarray(o["points"], float) + np.asarray(c, float)
                fcs = np.asarray(o["faces"], int)
                g = UsdGeom.Mesh.Define(stage, f"{path}/mesh_{i}")
                g.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in pts])
                g.CreateFaceVertexCountsAttr([3] * len(fcs))
                g.CreateFaceVertexIndicesAttr([int(v) for v in fcs.reshape(-1)])
                g.CreateSubdivisionSchemeAttr("none")
                nrm = np.zeros_like(pts)
                tri = pts[fcs]
                fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                for corner in range(3):
                    np.add.at(nrm, fcs[:, corner], fn)
                nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                g.CreateNormalsAttr([Gf.Vec3f(*map(float, v)) for v in nrm])
                g.SetNormalsInterpolation("vertex")
                prim = g.GetPrim()
                if o.get("collide", True):
                    UsdPhysics.CollisionAPI.Apply(prim)
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
                    col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    col.CreateContactOffsetAttr(0.002)
                    col.CreateRestOffsetAttr(0.0)
                    bind_physics_material(str(prim.GetPath()), mat, stage=stage)
                bind_visual_material(str(prim.GetPath()), look(k), stage=stage)
                if not o.get("visible", True):  # a collider only: the part is drawn by another (visual) mesh
                    UsdGeom.Imageable(prim).MakeInvisible()
                continue
            if o.get("shape") == "cyl":
                ax = o.get("axis", "Z")
                g = UsdGeom.Cylinder.Define(stage, f"{path}/cyl_{i}")
                g.CreateAxisAttr(ax)
                i_len = "XYZ".index(ax)
                g.CreateHeightAttr(float(s[i_len]))
                g.CreateRadiusAttr(float(max(v for j, v in enumerate(s) if j != i_len)) / 2)
                g.AddTranslateOp().Set(Gf.Vec3d(*c))
                prim = g.GetPrim()
            else:
                cube = UsdGeom.Cube.Define(stage, f"{path}/box_{i}")
                cube.CreateSizeAttr(1.0)
                cube.AddTranslateOp().Set(Gf.Vec3d(*c))
                cube.AddScaleOp().Set(Gf.Vec3f(*s))
                prim = cube.GetPrim()
            if o.get("collide", True):
                UsdPhysics.CollisionAPI.Apply(prim)
                col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                col.CreateContactOffsetAttr(0.002)
                col.CreateRestOffsetAttr(0.0)
                bind_physics_material(str(prim.GetPath()), mat, stage=stage)
            bind_visual_material(str(prim.GetPath()), look(k), stage=stage)

    base = f"{prim_path}/body"
    body(base, (0, 0, 0), spec.base_mass)
    if spec.com is not None:
        UsdPhysics.MassAPI(stage.GetPrimAtPath(base)).CreateCenterOfMassAttr(Gf.Vec3f(*spec.com))
    shapes(base, spec.boxes)
    # place every part at its starting position (start_at), so the very first frame already shows the start state
    world = {None: np.zeros(3)}
    rot = {None: np.eye(3)}
    paths = {None: base}
    for s in spec.sliders:
        q0 = _START.get((spec.name, s.name), 0.0)
        ax = np.eye(3)["XYZ".index(s.axis)]
        Rp = rot[s.parent]
        w = world[s.parent] + Rp @ np.array(s.origin, float)
        if s.kind == "revolute":
            Rc = Rp @ _axis_rot(ax, q0)
        else:
            w = w + Rp @ (ax * q0)
            Rc = Rp
        world[s.name], rot[s.name] = w, Rc
        p = f"{prim_path}/{s.name}"
        paths[s.name] = p
        body(p, tuple(w.tolist()), s.mass, s.gravity, _quat(Rc))
        shapes(p, s.boxes)
        if s.kind == "revolute":
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{p}/{_j(s)}")
            lo, hi = (math.degrees(s.lock_at), math.degrees(s.lock_at) + 0.01) if s.lock_until else (math.degrees(s.low), math.degrees(s.high))
            j.CreateLowerLimitAttr(lo)
            j.CreateUpperLimitAttr(hi)
            drive_kind = "angular"
        else:
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{p}/{_j(s)}")
            lo, hi = (s.lock_at, s.lock_at + 1e-4) if s.lock_until else (s.low, s.high)
            j.CreateLowerLimitAttr(lo)
            j.CreateUpperLimitAttr(hi)
            drive_kind = "linear"
        j.CreateAxisAttr(s.axis)
        j.CreateBody0Rel().SetTargets([Sdf.Path(paths[s.parent])])
        j.CreateBody1Rel().SetTargets([Sdf.Path(p)])
        j.CreateLocalPos0Attr(Gf.Vec3f(*s.origin))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), drive_kind)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(s.stiffness)
        drive.CreateDampingAttr(s.damping)
        # start (and hold, until the first control step) at the part's starting position
        q0 = _START.get((spec.name, s.name), 0.0)
        unit = math.degrees if s.kind == "revolute" else float
        drive.CreateTargetPositionAttr(unit(q0))
        js = PhysxSchema.JointStateAPI.Apply(j.GetPrim(), drive_kind)
        js.CreatePositionAttr(unit(q0))
        js.CreateVelocityAttr(0.0)
    bp = stage.GetPrimAtPath(base)
    UsdPhysics.ArticulationRootAPI.Apply(bp)
    a = PhysxSchema.PhysxArticulationAPI.Apply(bp)
    a.CreateEnabledSelfCollisionsAttr(False)
    a.CreateSolverPositionIterationCountAttr(32)
    a.CreateSolverVelocityIterationCountAttr(4)
    if not spec.fixed:
        return root.GetPrim()
    anchor = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/anchor")
    anchor.CreateBody1Rel().SetTargets([Sdf.Path(base)])
    t = UsdGeom.Xformable(bp).ComputeLocalToWorldTransform(0)
    wh, ro = t.ExtractTranslation(), t.ExtractRotationQuat()
    anchor.CreateLocalPos0Attr(Gf.Vec3f(*wh))
    anchor.CreateLocalRot0Attr(Gf.Quatf(ro.GetReal(), *ro.GetImaginary()))
    anchor.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    anchor.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    return root.GetPrim()


def _arena(spec):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class FxCfg(SpawnerCfg):
        func: object = clone(_spawn)
        fx_name: str = ""

    class FxObject(Object):
        def _generate_articulation_cfg(self):
            act = {s.name: ImplicitActuatorCfg(joint_names_expr=[_j(s)], stiffness=s.stiffness, damping=s.damping,
                                               friction=s.friction, dynamic_friction=s.dynamic_friction)
                   for s in spec.sliders}
            cfg = self._add_initial_pose_to_cfg(ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg,
                                                                actuators=act))
            if not spec.sliders:   # a jointless free body: no joint regexes to match
                cfg.init_state.joint_pos = {}; cfg.init_state.joint_vel = {}
            if any((spec.name, s.name) in _START for s in spec.sliders):
                cfg.init_state.joint_pos = {_j(s): _START.get((spec.name, s.name), 0.0) for s in spec.sliders}
            return cfg

    return FxObject(name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=FxCfg(fx_name=spec.name),
                    initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=tuple(spec.quat)))


def _np(x):
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _state(env, spec):
    import torch

    base = env.unwrapped
    asset = base.scene[spec.name]
    names = list(asset.joint_names)
    q, qd = _np(asset.data.joint_pos)[0], _np(asset.data.joint_vel)[0]
    ids = [names.index(_j(s)) for s in spec.sliders]
    dt = 1.0 / 20.0
    if spec.name not in _STARTED:
        _STARTED.add(spec.name)
        start = torch.tensor([[_START.get((spec.name, s.name), 0.0) for s in spec.sliders]], device=base.device,
                             dtype=torch.float32)
        asset.write_joint_state_to_sim_index(position=start, velocity=torch.zeros_like(start), joint_ids=ids)
        for k, i in enumerate(ids):
            q[i], qd[i] = float(start[0, k]), 0.0
    Q = {s.name: float(q[names.index(_j(s))]) for s in spec.sliders}
    V = {s.name: float(qd[names.index(_j(s))]) for s in spec.sliders}
    for s in spec.sliders:
        if _BROKEN.get(s.name):
            continue
        hit = False
        if s.breaks_at is not None and abs(Q[s.name]) >= s.breaks_at:
            hit = True
        if s.slam_zone is not None and abs(Q[s.name]) <= s.slam_zone:
            prev = _PREV.get(s.name, Q[s.name])
            est = max(abs(V[s.name]), abs(prev - Q[s.name]) / dt)
            if est > s.slam_speed:
                hit = True
        if hit:
            _BROKEN[s.name] = True
            if s.breaks_at is not None:
                i = names.index(_j(s))
                asset.write_joint_stiffness_to_sim_index(stiffness=0.0, joint_ids=[i])
                asset.write_joint_damping_to_sim_index(damping=s.limp_damping, joint_ids=[i])
                if s.broken_low is not None:
                    lim = torch.tensor([[[s.broken_low, s.high]]], device=base.device, dtype=torch.float32)
                    asset.write_joint_position_limit_to_sim_index(limits=lim, joint_ids=[i],
                                                                  warn_limit_violation=False)
    for s in spec.sliders:
        if s.lock_until and not _UNLOCKED.get(s.name) and ((Q[s.lock_until[0]] >= s.lock_until[1]) if s.lock_until[1] >= 0 else (Q[s.lock_until[0]] <= s.lock_until[1])):
            _UNLOCKED[s.name] = True
            i = names.index(_j(s))
            lim = torch.tensor([[[s.low, s.high]]], device=base.device, dtype=torch.float32)
            asset.write_joint_position_limit_to_sim_index(limits=lim, joint_ids=[i], warn_limit_violation=False)
    for s in spec.sliders:
        if s.latch_at is not None and not _LATCHED.get(s.name) and ((Q[s.name] >= s.latch_at) if s.latch_at >= 0 else (Q[s.name] <= s.latch_at)):
            _LATCHED[s.name] = True
    for s in spec.sliders:
        if s.free_when and not _FREED.get(s.name) and all(_LATCHED.get(n) for n in s.free_when):
            _FREED[s.name] = True
            i = names.index(_j(s))
            lim = torch.tensor([[[s.free_limits[0], s.free_limits[1]]]], device=base.device, dtype=torch.float32)
            asset.write_joint_position_limit_to_sim_index(limits=lim, joint_ids=[i], warn_limit_violation=False)
    for s in spec.sliders:
        if s.relimit and not _RELIMITED.get(s.name) and abs(Q[s.relimit[0]]) >= s.relimit[1]:
            _RELIMITED[s.name] = True
            i = names.index(_j(s))
            lim = torch.tensor([[[s.relimit[2], s.relimit[3]]]], device=base.device, dtype=torch.float32)
            asset.write_joint_position_limit_to_sim_index(limits=lim, joint_ids=[i], warn_limit_violation=False)
    for s in spec.sliders:
        _PREV[s.name] = Q[s.name]
    targets = []
    for s in spec.sliders:
        t = s.target
        if s.retarget:
            if not _RETARGETED.get(s.name):
                v = Q[s.retarget[0]]
                if (v >= s.retarget[1]) if s.retarget[1] >= 0 else (v <= s.retarget[1]):
                    _RETARGETED[s.name] = True
            if _RETARGETED.get(s.name):
                t = s.retarget[2]
        if isinstance(s.follows, tuple):   # several leaders: ((name, ratio), ...)
            t = s.target + sum(r * Q[n] for n, r in s.follows)
        elif s.follows is not None:
            t = s.target + s.ratio * Q[s.follows] + s.rate_gain * V[s.follows]
        if s.drive is not None:
            v, part, val = s.drive
            if (Q[part] >= val) if val >= 0 else (Q[part] <= val):
                _DSTOP[s.name] = True
            cur = _DRV.setdefault(s.name, _START.get((spec.name, s.name), s.target))
            if not _DSTOP.get(s.name):
                cur = min(max(cur + v * dt, s.low), s.high)
                _DRV[s.name] = cur
            t = cur
        if s.when_broken is not None and _BROKEN.get(s.when_broken[0]):
            t = s.when_broken[1]
            if s.broken_stiffness is not None and not _RESTIFFED.get(s.name):
                _RESTIFFED[s.name] = True
                asset.write_joint_stiffness_to_sim_index(stiffness=s.broken_stiffness, joint_ids=[names.index(_j(s))])
        targets.append(t)
    tt = torch.tensor([targets], device=base.device, dtype=torch.float32)
    asset.set_joint_position_target_index(target=tt, joint_ids=ids)
    asset.set_joint_velocity_target_index(target=torch.zeros_like(tt), joint_ids=ids)
    origin = _np(base.scene.env_origins)[0]
    bp = _np(asset.data.body_pos_w)[0] - origin
    bodies = {n: bp[k] for k, n in enumerate(asset.body_names)}
    if spec.fixed:
        p, quat, lv = np.array(spec.pos), np.array([0, 0, 0, 1.0]), np.zeros(3)
    else:
        p = _np(asset.data.root_pos_w)[0] - origin
        quat, lv = _np(asset.data.root_quat_w)[0], _np(asset.data.root_lin_vel_w)[0]  # Isaac Lab 3: (x, y, z, w)
    av = np.zeros(3) if spec.fixed else _np(asset.data.root_ang_vel_w)[0]
    return FixtureState(pos=p, quat_xyzw=quat, lin_vel=lv, ang_vel=av, low=p, high=p, q=Q, speed=V,
                        broken={s.name: bool(_BROKEN.get(s.name)) for s in spec.sliders}, bodies=bodies,
                        latched={s.name: bool(_LATCHED.get(s.name)) for s in spec.sliders if s.latch_at is not None},
                        hands=_hands(base))


def _hands(base) -> dict:
    """Where each gripper's tool point is (world, like the task's positions) and how far its fingers are open."""
    from yam_arena.constants import FINGER_JOINTS, FINGER_OPEN
    from yam_arena.runtime import pad_tips_world, tcp_world

    origin = _np(base.scene.env_origins)[0]
    out = {}
    for arm in ("left_robot", "right_robot"):
        if arm not in base.scene.keys():
            continue
        robot = base.scene[arm]
        fingers = robot.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        opening = float(_np(robot.data.joint_pos)[0][fingers].mean() / FINGER_OPEN)
        out[arm.split("_")[0]] = {"tcp": np.asarray(tcp_world(robot), dtype=float).reshape(-1)[:3] - origin,
                                  "tips": np.asarray(pad_tips_world(robot), dtype=float).reshape(-1)[:3] - origin,
                                  "open": float(np.clip(opening, 0.0, 1.0))}
    return out


_START: dict = {}
_DRV: dict = {}
_DSTOP: dict = {}
_RELIMITED: dict = {}
_RETARGETED: dict = {}
_RESTIFFED: dict = {}
_LATCHED: dict = {}
_FREED: dict = {}


def start_at(fixture_name, part, value):
    """Start a part away from 0 (e.g. a lid that starts open)."""
    _START[(fixture_name, part)] = value


def _patch(scene):
    if getattr(scene, "_kit3", False):
        return
    bt, bs = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Fixture):
            _SPECS[spec.name] = spec
            return _arena(spec)
        return bt(spec)

    def object_states(env, names):
        plain = [n for n in names if n not in _SPECS]
        out = bs(env, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                out[n] = sys.modules[__name__]._state(env, _SPECS[n])
        return {n: out[n] for n in names}

    scene.task_object, scene.object_states, scene._kit3 = task_object, object_states, True
    _patch_reset()


def _apply_start(env):
    """Put every kit fixture's parts at their starting positions (and hold them there) right after a reset, so the
    settling steps, the first observation and the first frame already show the start state."""
    import torch

    base = getattr(env, "unwrapped", env)
    for name, spec in _SPECS.items():
        if name not in base.scene.keys():
            continue
        asset = base.scene[name]
        names = list(asset.joint_names)
        ids = [names.index(_j(s)) for s in spec.sliders]
        start = torch.tensor([[_START.get((name, s.name), 0.0) for s in spec.sliders]], device=base.device, dtype=torch.float32)
        asset.write_joint_state_to_sim_index(position=start, velocity=torch.zeros_like(start), joint_ids=ids)
        asset.set_joint_position_target_index(target=start, joint_ids=ids)
        asset.set_joint_velocity_target_index(target=torch.zeros_like(start), joint_ids=ids)


def _patch_reset():
    from isaaclab.envs import ManagerBasedEnv

    for cls in {ManagerBasedEnv, *ManagerBasedEnv.__subclasses__()}:
        if "reset" in cls.__dict__ and not getattr(cls.reset, "_kit3", False):
            orig = cls.reset

            def reset(self, *a, _orig=orig, **k):
                out = _orig(self, *a, **k)
                _apply_start(self)
                return out
            reset._kit3 = True
            cls.reset = reset


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "yam_arena.scene":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        ex = spec.loader.exec_module

        def go(m):
            ex(m)
            _patch(m)

        spec.loader.exec_module = go
        return spec


if "yam_arena.scene" in sys.modules:
    _patch(sys.modules["yam_arena.scene"])
elif not any(isinstance(f, _Hook) for f in sys.meta_path):
    sys.meta_path.insert(0, _Hook())
