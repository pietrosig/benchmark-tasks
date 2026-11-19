"""foodkit: multi-shape rigid bodies and a few runtime joints for a yam_arena task file (one per task folder).

yam_arena's task-file objects are single primitives. Here a :class:`Body` is one rigid body made of several boxes,
cylinders and spheres (some visual only), with its own friction; ``kinematic`` bodies stay where they are put. Two
kinds of joint can be added between bodies, both authored when the simulator builds the scene:

- :func:`glue`: a fixed joint holding two bodies as they start; the task file switches it off (:func:`set_glue`) once
  it has decided they part (a cut finished through it). Every reset switches it back on.
- :func:`grip`: a D6 joint from one body to another, authored switched off. :func:`grip_on` switches it on at the two
  bodies' current relative pose (it holds them there), with one free slide along a chosen axis that a capped drive
  holds at 0 (a fork's tines held in food until pulled out harder than the cap); :func:`grip_off` lets go.

The kit hooks ``yam_arena.scene`` (spawning the bodies, reporting their poses) and the embodiment's reset and control
step (:func:`on_reset`, :func:`on_step`), sets forces on bodies (:func:`set_force`), reports which robot links touch
chosen bodies (:func:`robot_contacts`) and lets a demo waypoint carry a full 3x3 grip rotation
(:func:`allow_matrix_grips`). Pure Python at import time: everything that needs Isaac Sim is imported lazily.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass

import numpy as np
from yam_arena.task_file import Box, ObjectState

Color = tuple


@dataclass(frozen=True)
class Shape:
    """A box (``kind`` "box", ``size`` x, y, z), a cylinder ("cyl": ``size`` = (radius, length), along ``axis``) or a
    sphere ("sphere": ``size`` = (radius,)), centred at ``centre`` in the body frame, optionally rotated by ``quat``
    (x, y, z, w)."""

    kind: str
    centre: tuple
    size: tuple
    color: Color = (0.6, 0.6, 0.6)
    collide: bool = True
    axis: str = "Z"
    quat: tuple = (0.0, 0.0, 0.0, 1.0)

    def half_extent(self) -> np.ndarray:
        if self.kind == "box":
            h = np.array(self.size, float) / 2
        elif self.kind == "cyl":
            h = np.full(3, self.size[0])
            h["XYZ".index(self.axis)] = self.size[1] / 2
        else:
            h = np.full(3, self.size[0])
        return np.abs(rotation(self.quat)) @ h


def box(centre, size, color=(0.6, 0.6, 0.6), collide=True, quat=(0.0, 0.0, 0.0, 1.0)) -> Shape:
    return Shape("box", tuple(centre), tuple(size), tuple(color), collide, "Z", tuple(quat))


def cyl(centre, radius, length, axis="Z", color=(0.6, 0.6, 0.6), collide=True) -> Shape:
    return Shape("cyl", tuple(centre), (radius, length), tuple(color), collide, axis)


def rotation(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_of(m) -> tuple:
    """(x, y, z, w) of a rotation matrix."""
    m = np.asarray(m, float)
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (x / n, y / n, z / n, w / n)


@dataclass(frozen=True)
class Body(Box):
    """A rigid body made of ``shapes`` (subclasses Box so yam_arena's checks accept it). ``pos``: the body origin
    (world, m); ``quat0``: its starting orientation (x, y, z, w). ``friction``: (static, dynamic); ``slick``: contacts
    take the lower friction of the pair. ``ignore``: bodies earlier in OBJECTS it never collides with."""

    shapes: tuple = ()
    kinematic: bool = False
    friction: tuple = (0.6, 0.5)
    slick: bool = False
    damping: tuple = (0.0, 0.05)
    com: tuple | None = None
    quat0: tuple = (0.0, 0.0, 0.0, 1.0)
    ignore: tuple = ()
    iterations: int = 32

    @staticmethod
    def build(name, pos, mass, shapes, **kw) -> "Body":
        lo = np.min([np.array(s.centre) - s.half_extent() for s in shapes], axis=0)
        hi = np.max([np.array(s.centre) + s.half_extent() for s in shapes], axis=0)
        return Body(name=name, pos=tuple(float(v) for v in pos), size=tuple((hi - lo).tolist()), mass=mass,
                    shapes=tuple(shapes), **kw)

    def bounds(self):
        lo = np.min([np.array(s.centre) - s.half_extent() for s in self.shapes], axis=0)
        hi = np.max([np.array(s.centre) + s.half_extent() for s in self.shapes], axis=0)
        return lo, hi


@dataclass(frozen=True)
class BodyState(ObjectState):
    """``pos``/``quat_xyzw``: the body origin's pose (world, m)."""

    def rot(self) -> np.ndarray:
        return rotation(self.quat_xyzw)


def state_of(spec: Body, pos, q, v=(0, 0, 0), w=(0, 0, 0)) -> BodyState:
    lo, hi = spec.bounds()
    c = np.array([[a, b, d] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for d in (lo[2], hi[2])])
    pts = np.asarray(pos, float) + c @ rotation(q).T
    return BodyState(pos=np.asarray(pos, float), quat_xyzw=np.asarray(q, float), lin_vel=np.asarray(v, float),
                     ang_vel=np.asarray(w, float), low=pts.min(0), high=pts.max(0))


# -- joints --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Glue:
    name: str
    body0: str
    body1: str
    anchor: tuple  # world, at the start


@dataclass(frozen=True)
class Grip:
    name: str
    body0: str  # the holder (a fork)
    body1: str  # the held (a slab of food)
    slide_cap: float  # N: the drive holding the slide at 0 gives way above this (pulled out)
    slide_max: float  # m: how far the slide may run out


_SPECS: dict = {}
_GLUES: list = []
_GRIPS: list = []
_PATHS: dict = {}
_ATTR: dict = {}


def glue(name, body0, body1, anchor) -> Glue:
    g = Glue(name, body0, body1, tuple(anchor))
    _GLUES.append(g)
    return g


def grip(name, body0, body1, slide_cap, slide_max) -> Grip:
    g = Grip(name, body0, body1, float(slide_cap), float(slide_max))
    _GRIPS.append(g)
    return g


def _stage():
    from isaaclab.sim.utils.stage import get_current_stage
    return get_current_stage()


def _author_joints(stage, prim_path: str, spec: Body) -> None:
    from pxr import Gf, UsdPhysics

    parent = prim_path.rsplit("/", 1)[0]
    for g in _GLUES:
        if g.body1 != spec.name:
            continue
        b0 = _SPECS[g.body0]
        j = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/{g.name}")
        j.CreateBody0Rel().SetTargets([f"{parent}/{g.body0}/body"])
        j.CreateBody1Rel().SetTargets([f"{prim_path}/body"])
        r0, r1 = rotation(b0.quat0), rotation(spec.quat0)
        a = np.asarray(g.anchor, float)
        j.CreateLocalPos0Attr(Gf.Vec3f(*(r0.T @ (a - np.asarray(b0.pos))).tolist()))
        j.CreateLocalPos1Attr(Gf.Vec3f(*(r1.T @ (a - np.asarray(spec.pos))).tolist()))
        q0, q1 = quat_of(r0.T), quat_of(r1.T)  # the world frame seen from each body
        j.CreateLocalRot0Attr(Gf.Quatf(q0[3], *q0[:3]))
        j.CreateLocalRot1Attr(Gf.Quatf(q1[3], *q1[:3]))
        j.CreateJointEnabledAttr(True)
        j.CreateExcludeFromArticulationAttr(True)
        _PATHS[g.name] = f"{prim_path}/{g.name}"
    for g in _GRIPS:
        if g.body1 != spec.name:
            continue
        j = UsdPhysics.Joint.Define(stage, f"{prim_path}/{g.name}")  # a plain Joint is a D6
        j.CreateBody0Rel().SetTargets([f"{parent}/{g.body0}/body"])
        j.CreateBody1Rel().SetTargets([f"{prim_path}/body"])
        j.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateJointEnabledAttr(False)
        j.CreateCollisionEnabledAttr(False)
        j.CreateExcludeFromArticulationAttr(True)
        prim = j.GetPrim()
        for dof in ("transY", "transZ", "rotX", "rotY", "rotZ"):
            lim = UsdPhysics.LimitAPI.Apply(prim, dof)  # low > high: locked
            lim.CreateLowAttr(1.0)
            lim.CreateHighAttr(-1.0)
        lim = UsdPhysics.LimitAPI.Apply(prim, "transX")
        lim.CreateLowAttr(-0.0005)
        lim.CreateHighAttr(float(g.slide_max))
        d = UsdPhysics.DriveAPI.Apply(prim, "transX")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(4000.0)
        d.CreateDampingAttr(20.0)
        d.CreateTargetPositionAttr(0.0)
        d.CreateMaxForceAttr(float(g.slide_cap))
        _PATHS[g.name] = f"{prim_path}/{g.name}"


def _joint_attr(name, attr):
    key = (name, attr)
    if key not in _ATTR:
        prim = _stage().GetPrimAtPath(_PATHS[name])
        _ATTR[key] = prim.GetAttribute(attr)
    return _ATTR[key]


def set_glue(name: str, on: bool) -> None:
    _joint_attr(name, "physics:jointEnabled").Set(bool(on))


def grip_on(name: str, pos0, rot0, pos1, rot1) -> None:
    """Switch grip ``name`` on with its frame at ``pos0``/``rot0`` in body0's frame and ``pos1``/``rot1`` in body1's
    (the same world frame seen from each: the bodies are held as they are). The frame's x axis is the free slide."""
    from pxr import Gf

    q0, q1 = quat_of(rot0), quat_of(rot1)
    _joint_attr(name, "physics:localPos0").Set(Gf.Vec3f(*[float(v) for v in pos0]))
    _joint_attr(name, "physics:localRot0").Set(Gf.Quatf(q0[3], *q0[:3]))
    _joint_attr(name, "physics:localPos1").Set(Gf.Vec3f(*[float(v) for v in pos1]))
    _joint_attr(name, "physics:localRot1").Set(Gf.Quatf(q1[3], *q1[:3]))
    _joint_attr(name, "physics:jointEnabled").Set(True)


def grip_off(name: str) -> None:
    _joint_attr(name, "physics:jointEnabled").Set(False)


def _joints_reset() -> None:
    for g in _GLUES:  # off and on: rebuilt whole from the bodies' reset poses
        set_glue(g.name, False)
        set_glue(g.name, True)
    for g in _GRIPS:
        grip_off(g.name)


# -- spawning ------------------------------------------------------------------------------------------------------

def _spawn(prim_path, cfg, translation=None, orientation=None, **kw):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    from yam_arena.station import linear_rgb

    spec = _SPECS[cfg.body_name]
    stage = _stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    phys = f"{prim_path}/Looks/physics"
    m = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction[0], dynamic_friction=spec.friction[1],
                                       friction_combine_mode="min" if spec.slick else "average")
    m.func(phys, m)
    looks: dict = {}

    def look(c):
        c = tuple(c)
        if c not in looks:
            p = f"{prim_path}/Looks/c{len(looks)}"
            s = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(c), roughness=0.35)
            s.func(p, s)
            looks[c] = p
        return looks[c]

    bp = f"{prim_path}/body"
    xf = UsdGeom.Xform.Define(stage, bp)
    xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    prim = xf.GetPrim()
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    if spec.kinematic:
        rb.CreateKinematicEnabledAttr(True)
    ma = UsdPhysics.MassAPI.Apply(prim)
    ma.CreateMassAttr(float(spec.mass))
    if spec.com is not None:
        ma.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in spec.com]))
    if spec.ignore:
        parent = prim_path.rsplit("/", 1)[0]
        rel = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
        for o in spec.ignore:
            rel.AddTarget(f"{parent}/{o}/body")
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(int(spec.iterations))
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateLinearDampingAttr(float(spec.damping[0]))
    px.CreateAngularDampingAttr(float(spec.damping[1]))
    for i, s in enumerate(spec.shapes):
        if s.kind == "box":
            g = UsdGeom.Cube.Define(stage, f"{bp}/s{i}")
            g.CreateSizeAttr(1.0)
            g.AddTranslateOp().Set(Gf.Vec3d(*s.centre))
            if tuple(s.quat) != (0.0, 0.0, 0.0, 1.0):
                g.AddOrientOp().Set(Gf.Quatf(s.quat[3], *s.quat[:3]))
            g.AddScaleOp().Set(Gf.Vec3f(*s.size))
        elif s.kind == "cyl":
            g = UsdGeom.Cylinder.Define(stage, f"{bp}/s{i}")
            g.CreateAxisAttr(s.axis)
            g.CreateRadiusAttr(float(s.size[0]))
            g.CreateHeightAttr(float(s.size[1]))
            g.AddTranslateOp().Set(Gf.Vec3d(*s.centre))
        else:
            g = UsdGeom.Sphere.Define(stage, f"{bp}/s{i}")
            g.CreateRadiusAttr(float(s.size[0]))
            g.AddTranslateOp().Set(Gf.Vec3d(*s.centre))
        p = str(g.GetPrim().GetPath())
        bind_visual_material(p, look(s.color), stage=stage)
        if s.collide:
            UsdPhysics.CollisionAPI.Apply(g.GetPrim())
            c = PhysxSchema.PhysxCollisionAPI.Apply(g.GetPrim())
            c.CreateContactOffsetAttr(0.0015)
            c.CreateRestOffsetAttr(0.0)
            bind_physics_material(p, phys, stage=stage)
    _author_joints(stage, prim_path, spec)
    return root.GetPrim()


def _arena_object(spec: Body):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class FoodSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn)
        body_name: str = ""

    return Object(name=spec.name, object_type=ObjectType.RIGID, spawner_cfg=FoodSpawnerCfg(body_name=spec.name),
                  initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=tuple(spec.quat0)))


def as_np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def pose(base, name):
    """(position, rotation matrix, linear velocity, angular velocity) of a body, world frame."""
    d = base.scene[name].data
    o = as_np(base.scene.env_origins[0])
    p = as_np(d.root_link_pose_w)[0]
    v = as_np(d.root_com_vel_w)[0] if hasattr(d, "root_com_vel_w") else as_np(d.root_vel_w)[0]
    return p[:3] - o, rotation(p[3:7]), v[:3], v[3:6]


def _body_state(env, spec):
    base = env.unwrapped
    d = base.scene[spec.name].data
    o = as_np(base.scene.env_origins[0])
    p = as_np(d.root_link_pose_w)[0]
    v = as_np(d.root_com_vel_w)[0] if hasattr(d, "root_com_vel_w") else as_np(d.root_vel_w)[0]
    return state_of(spec, p[:3] - o, p[3:7], v[:3], v[3:6])


def set_force(base, name, f_world, at_local=None) -> None:
    """Apply a constant force (world frame) to a body until it is set again, at ``at_local`` (its frame) or its origin."""
    import torch

    asset = base.scene[name]
    _, r, _, _ = pose(base, name)
    f = torch.tensor(r.T @ np.asarray(f_world, float), dtype=torch.float32, device=asset.device).view(1, 1, 3)
    at = None if at_local is None else torch.tensor(np.asarray(at_local, float), dtype=torch.float32,
                                                     device=asset.device).view(1, 1, 3)
    asset.permanent_wrench_composer.set_forces_and_torques(forces=f, torques=torch.zeros_like(f), positions=at,
                                                           is_global=False)


# -- robot contacts (PhysX contact view from every link of both arms to chosen bodies) -----------------------------

_CONTACT: dict = {}
CONTACT_N = 0.02


def _host(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    import warp as wp
    return wp.to_torch(x).cpu().numpy()


def robot_contacts(base, names) -> list | None:
    """Robot links ("arm/link") touching any of the bodies ``names`` at the last physics step; None if PhysX can't say."""
    key = tuple(names)
    v = _CONTACT.get(key)
    if v is None:
        try:
            from isaaclab_physx.physics import PhysxManager
            from pxr import Usd, UsdPhysics

            stage = _stage()
            env0 = base.scene.env_prim_paths[0]
            links, sensors = [], []
            for arm in ("left_robot", "right_robot"):
                root = f"{env0}/{base.scene[arm].cfg.prim_path.rstrip('/').rsplit('/', 1)[-1]}"
                for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
                    if p.HasAPI(UsdPhysics.RigidBodyAPI) and "PhysxContactReportAPI" in p.GetAppliedSchemas():
                        links.append(f"{arm.split('_')[0]}/{p.GetName()}")
                        sensors.append(p.GetPath().pathString)
            filters = [f"{env0}/{base.scene[n].cfg.prim_path.rstrip('/').rsplit('/', 1)[-1]}/body" for n in names]
            view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
                sensors, filter_patterns=[filters] * len(sensors), max_contact_data_count=0)
            dt = float(getattr(base, "physics_dt", None) or base.sim.get_physics_dt())
            v = _CONTACT[key] = (view, links, len(filters), dt)
            print(f"[foodkit] contact view: {len(sensors)} robot links x {len(filters)} bodies", flush=True)
        except Exception as e:  # noqa: BLE001
            v = _CONTACT[key] = f"{type(e).__name__}: {e}"
            print(f"[foodkit] contact view failed: {v}", flush=True)
    if isinstance(v, str):
        return None
    view, links, nf, dt = v
    f = np.linalg.norm(_host(view.get_contact_force_matrix(dt=dt)).reshape(len(links), nf, 3), axis=-1)
    return [links[i] for i in range(len(links)) if f[i].max() > CONTACT_N]


# -- hooks -----------------------------------------------------------------------------------------------------------

_RESET: list = []
_STEP: list = []


def on_reset(fn):
    _RESET.append(fn)
    return fn


def on_step(fn):
    _STEP.append(fn)
    return fn


def _patch_scene(scene) -> None:
    if getattr(scene, "_foodkit", False):
        return
    to, st = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Body):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return to(spec)

    def object_states(env, names):
        plain = [n for n in names if n not in _SPECS]
        out = st(env, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                out[n] = _body_state(env, _SPECS[n])
        return {n: out[n] for n in names}

    scene.task_object, scene.object_states, scene._foodkit = task_object, object_states, True


def _patch_embodiment(module) -> None:
    cls = getattr(module, "YamArenaEmbodiment", None)
    if cls is None or getattr(cls, "_foodkit", False):
        return
    br, bs = cls.reset, cls._task_step

    def reset(self, *a, **k):
        for g in _GRIPS:
            if g.name in _PATHS:
                grip_off(g.name)
        obs = br(self, *a, **k)
        if _PATHS:
            _joints_reset()
        for fn in _RESET:
            fn()
        return obs

    def _task_step(self, *a, **k):
        for fn in _STEP:
            fn(self._env)
        return bs(self, *a, **k)

    cls.reset, cls._task_step, cls._foodkit = reset, _task_step, True


_TARGETS = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        ex = spec.loader.exec_module

        def exec_module(m):
            ex(m)
            _TARGETS[fullname](m)

        spec.loader.exec_module = exec_module
        return spec


def allow_matrix_grips() -> None:
    """Let a demo Waypoint's ``yaw`` be a full 3x3 TCP rotation (world frame)."""
    from yam_arena import demo

    if getattr(demo, "_foodkit_grips", False):
        return
    plain = demo.grip_rotation

    def grip_rotation(yaw=0.0):
        return np.asarray(yaw, float) if np.ndim(yaw) == 2 else plain(yaw)

    demo.grip_rotation = grip_rotation
    demo._foodkit_grips = True


def install() -> None:
    for name, patch in _TARGETS.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
