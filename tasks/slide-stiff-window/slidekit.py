"""slidekit: a body fixed to the table with sliding parts, for yam_arena task files.

yam_arena's own objects are loose rigid bodies. A ``Fixture`` is an articulation: a base made of boxes, fixed to the
table, with ``Slider`` parts on prismatic joints (along x, y or z), which may hang off each other in series. A slider can
have a spring, damping and joint friction, and a break point: once it has been pulled ``breaks_at`` from rest its
spring goes limp for good and its travel opens up to ``broken_low``. Importing this module hooks ``yam_arena.scene``
when it is imported, without modifying yam_arena; the task's ``score()`` then gets a ``FixtureState`` for the fixture,
with each slider's position (``q``), speed and whether it has broken.

Pure Python at import time; everything that needs Isaac Sim is imported lazily. One fixture state per process: a
broken slider stays broken until the process ends (one trial per ``inspect-robots run``).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena.task_file import Box, ObjectState

__all__ = ["Fixture", "FixtureState", "Slider", "fixture"]

Color = tuple  # sRGB, or sRGBA for see-through parts
BoxSpec = tuple  # (centre, full size (m), colour) in the part's frame


@dataclass(frozen=True)
class Slider:
    name: str
    origin: tuple  # rest position in the parent's frame (the parent is the base unless ``parent`` is set)
    axis: str  # "X", "Y" or "Z"
    boxes: tuple  # BoxSpec, in the slider's frame
    low: float  # travel limits (m)
    high: float
    parent: str | None = None
    stiffness: float = 0.0  # N/m towards rest
    damping: float = 1.0  # N s/m
    friction: float = 0.0
    mass: float = 0.1  # kg
    breaks_at: float | None = None  # |q| (m) at which it fails
    binding: tuple | None = None  # (static N, dynamic N): dry joint friction, applied by PhysX every physics step: the
    # slider stays put until pushed or pulled along its axis with more than the static effort, then drags with the dynamic one
    broken_low: float | None = None  # its lower limit after failing
    limp_damping: float = 0.5


@dataclass(frozen=True)
class Fixture(Box):
    boxes: tuple = ()  # BoxSpec of the fixed base, in the fixture's frame
    sliders: tuple = ()
    friction: float = 0.8


@dataclass(frozen=True)
class FixtureState(ObjectState):
    q: dict = field(default_factory=dict)  # slider name -> position (m) from rest
    speed: dict = field(default_factory=dict)  # slider name -> m/s
    broken: dict = field(default_factory=dict)  # slider name -> bool
    hands: dict = field(default_factory=dict)  # arm -> {"tcp": world xyz between the fingertips, "open": 0 closed .. 1 open}
    drive: dict = field(default_factory=dict)  # slider name -> force (N) its spring/friction drive puts on it, along its axis
    accel: dict = field(default_factory=dict)  # slider name -> acceleration (m/s^2) over the last control step
    contacts: dict | None = None  # part ("body" or a slider's name) -> robot links touching it now, as "arm/link";
    # None if PhysX could not report contacts


def fixture(name, pos, boxes, sliders, color=(0.7, 0.7, 0.7)) -> Fixture:
    """A fixture whose frame sits at ``pos`` (world, m), unrotated."""
    return Fixture(name=name, pos=tuple(pos), size=(0.1, 0.1, 0.1), color=color, boxes=tuple(boxes),
                   sliders=tuple(sliders))


_SPECS: dict = {}
_STEP: dict = {}
_VPREV: dict = {}
_ACCEL: dict = {}
_BROKEN: dict = {}
_STARTED: set = set()
_CONTACT: dict = {}  # fixture name -> (contact view, robot links, parts, physics dt), or the error that stopped it
CONTACT_N = 0.01  # N: a contact force above this between a robot link and a part of the fixture counts as touching


def _joint(s: Slider) -> str:
    return f"{s.name}_slide"


def _spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
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
    looks: dict = {}

    def look(c):
        if c not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            glass = len(c) == 4
            s = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(c[:3]), roughness=0.05 if glass else 0.45,
                                            metallic=0.0, opacity=c[3] if glass else 1.0)
            s.func(path, s)
            looks[c] = path
        return looks[c]

    def body(path, pos, mass):
        x = UsdGeom.Xform.Define(stage, path)
        x.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
        x.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
        p = x.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(p)
        UsdPhysics.MassAPI.Apply(p).CreateMassAttr(mass)
        b = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
        b.CreateMaxDepenetrationVelocityAttr(1.0)
        b.CreateSolverPositionIterationCountAttr(32)

    def boxes(path, items):
        for i, (c, s, k) in enumerate(items):
            cube = UsdGeom.Cube.Define(stage, f"{path}/box_{i}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*c))
            cube.AddScaleOp().Set(Gf.Vec3f(*s))
            g = cube.GetPrim()
            UsdPhysics.CollisionAPI.Apply(g)
            col = PhysxSchema.PhysxCollisionAPI.Apply(g)
            col.CreateContactOffsetAttr(0.002)
            col.CreateRestOffsetAttr(0.0)
            bind_physics_material(str(g.GetPath()), mat, stage=stage)
            bind_visual_material(str(g.GetPath()), look(k), stage=stage)

    base = f"{prim_path}/body"
    body(base, (0, 0, 0), 5.0)
    boxes(base, spec.boxes)
    world = {None: np.zeros(3)}
    paths = {None: base}
    for s in spec.sliders:
        w = world[s.parent] + np.array(s.origin)
        world[s.name] = w
        p = f"{prim_path}/{s.name}"
        paths[s.name] = p
        body(p, tuple(w.tolist()), s.mass)
        boxes(p, s.boxes)
        j = UsdPhysics.PrismaticJoint.Define(stage, f"{p}/{_joint(s)}")
        j.CreateAxisAttr(s.axis)
        j.CreateBody0Rel().SetTargets([Sdf.Path(paths[s.parent])])
        j.CreateBody1Rel().SetTargets([Sdf.Path(p)])
        j.CreateLocalPos0Attr(Gf.Vec3f(*s.origin))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLowerLimitAttr(s.low)
        j.CreateUpperLimitAttr(s.high)
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")  # without a drive PhysX applies no spring or damping
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(s.stiffness)
        drive.CreateDampingAttr(s.damping)
        drive.CreateTargetPositionAttr(0.0)
    bp = stage.GetPrimAtPath(base)
    UsdPhysics.ArticulationRootAPI.Apply(bp)
    a = PhysxSchema.PhysxArticulationAPI.Apply(bp)
    a.CreateEnabledSelfCollisionsAttr(True)
    a.CreateSolverPositionIterationCountAttr(32)
    a.CreateSolverVelocityIterationCountAttr(4)
    anchor = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/anchor")  # the base is fixed to the world where it spawns
    anchor.CreateBody1Rel().SetTargets([Sdf.Path(base)])
    t = UsdGeom.Xformable(bp).ComputeLocalToWorldTransform(0)
    where, rot = t.ExtractTranslation(), t.ExtractRotationQuat()
    anchor.CreateLocalPos0Attr(Gf.Vec3f(*where))
    anchor.CreateLocalRot0Attr(Gf.Quatf(rot.GetReal(), *rot.GetImaginary()))
    anchor.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    anchor.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    return root.GetPrim()


def _arena_object(spec: Fixture):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class FixtureSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn)
        fx_name: str = ""

    class FixtureObject(Object):
        def _generate_articulation_cfg(self):
            actuators = {s.name: ImplicitActuatorCfg(
                joint_names_expr=[_joint(s)], stiffness=s.stiffness, damping=s.damping,
                friction=s.binding[0] if s.binding else s.friction,
                dynamic_friction=s.binding[1] if s.binding else None) for s in spec.sliders}
            return self._add_initial_pose_to_cfg(ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg,
                                                                 actuators=actuators))

    return FixtureObject(name=spec.name, object_type=ObjectType.ARTICULATION,
                         spawner_cfg=FixtureSpawnerCfg(fx_name=spec.name),
                         initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0, 0, 0, 1)))


def _np(x):
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _state(env, spec: Fixture) -> FixtureState:
    import torch

    base = env.unwrapped
    asset = base.scene[spec.name]
    names = list(asset.joint_names)
    q, qd = _np(asset.data.joint_pos)[0], _np(asset.data.joint_vel)[0]
    ids = [names.index(_joint(s)) for s in spec.sliders]
    zeros = torch.zeros((1, len(ids)), device=base.device, dtype=torch.float32)
    if spec.name not in _STARTED:  # after the embodiment's reset: every slider at rest
        _STARTED.add(spec.name)
        asset.write_joint_state_to_sim_index(position=zeros, velocity=zeros, joint_ids=ids)
        q[ids], qd[ids] = 0.0, 0.0
    pos = {s.name: float(q[names.index(_joint(s))]) for s in spec.sliders}
    vel = {s.name: float(qd[names.index(_joint(s))]) for s in spec.sliders}
    # The force each slider's drive and friction put on it, as far as it is known: its spring and damping, plus, for a
    # binding slider that is sliding, its dynamic friction against the motion (while it sticks, its static friction
    # takes whatever is needed, up to the static effort, so it is not known and is reported as 0).
    step_dt = float(getattr(base, "step_dt", 0.05))
    step = _step_key(base, step_dt)
    drive = {}
    for s in spec.sliders:
        f = s.stiffness * (0.0 - pos[s.name]) - s.damping * vel[s.name]
        if s.binding is not None and abs(vel[s.name]) > 0.001:
            f -= np.sign(vel[s.name]) * s.binding[1]
        drive[s.name] = float(f)
    targets = [0.0 for _ in spec.sliders]
    if _STEP.get(spec.name) != step:
        dt_step = (step - _STEP[spec.name]) * step_dt if spec.name in _STEP else None
        for s in spec.sliders:
            if dt_step:
                _ACCEL[s.name] = (vel[s.name] - _VPREV.get(s.name, vel[s.name])) / dt_step
            _VPREV[s.name] = vel[s.name]
        _STEP[spec.name] = step
    tt = torch.tensor([targets], device=base.device, dtype=torch.float32)
    asset.set_joint_position_target_index(target=tt, joint_ids=ids)
    asset.set_joint_velocity_target_index(target=zeros, joint_ids=ids)
    for s in spec.sliders:
        if s.breaks_at is not None and abs(pos[s.name]) >= s.breaks_at and not _BROKEN.get(s.name):
            _BROKEN[s.name] = True
            i = names.index(_joint(s))
            asset.write_joint_stiffness_to_sim_index(stiffness=0.0, joint_ids=[i])
            asset.write_joint_damping_to_sim_index(damping=s.limp_damping, joint_ids=[i])
            if s.broken_low is not None:
                limits = torch.tensor([[[s.broken_low, s.high]]], device=base.device, dtype=torch.float32)
                asset.write_joint_position_limit_to_sim_index(limits=limits, joint_ids=[i], warn_limit_violation=False)
    p = np.array(spec.pos)
    return FixtureState(pos=p, quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=p,
                        high=p, q=pos, speed=vel, broken={s.name: bool(_BROKEN.get(s.name)) for s in spec.sliders},
                        hands=_hands(base), drive=drive, accel={s.name: _ACCEL.get(s.name, 0.0) for s in spec.sliders},
                        contacts=_robot_contacts(base, spec))


def _step_key(base, step_dt: float) -> int:
    """A number that goes up by one each control step: the env's step counter, or else the simulated time over the step."""
    c = getattr(base, "common_step_counter", None)
    if c is not None:
        return int(c)
    try:
        return int(round(float(base.sim.current_time) / step_dt))
    except Exception:  # noqa: BLE001
        return int(round(float(getattr(base.sim, "_current_time", 0.0)) / step_dt))


def _hands(base) -> dict:
    """Where each gripper's tool point is (world, like the task's positions) and how open its fingers are."""
    from yam_arena.constants import FINGER_JOINTS, FINGER_OPEN
    from yam_arena.runtime import tcp_world

    origin = _np(base.scene.env_origins)[0]
    out = {}
    for arm in ("left_robot", "right_robot"):
        if arm not in base.scene.keys():
            continue
        robot = base.scene[arm]
        fingers = robot.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        opening = float(_np(robot.data.joint_pos)[0][fingers].mean() / FINGER_OPEN)
        out[arm] = {"tcp": tcp_world(robot) - origin, "open": float(np.clip(opening, 0.0, 1.0))}
    return out


def _host(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    import warp as wp
    return wp.to_torch(x).cpu().numpy()


def _rigid_bodies(stage, root: str) -> list:
    from pxr import Usd, UsdPhysics
    return [p for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.RigidBodyAPI)]


def _contact_view(base, spec: Fixture):
    """A PhysX contact view from every link of both arms (the YAM spawns them with contact reporting on) to each rigid
    body of the fixture: its base ("body") and each slider."""
    from isaaclab.sim.utils.stage import get_current_stage
    from isaaclab_physx.physics import PhysxManager

    stage = get_current_stage()
    env0 = base.scene.env_prim_paths[0]

    def root(key):
        return f"{env0}/{base.scene[key].cfg.prim_path.rstrip('/').rsplit('/', 1)[-1]}"

    links, sensors = [], []
    for arm in ("left_robot", "right_robot"):
        if arm not in base.scene.keys():
            continue
        for p in _rigid_bodies(stage, root(arm)):
            if "PhysxContactReportAPI" in p.GetAppliedSchemas():
                links.append((arm, p.GetName()))
                sensors.append(p.GetPath().pathString)
    parts, filters = [], []
    for p in _rigid_bodies(stage, root(spec.name)):
        parts.append(p.GetName())
        filters.append(p.GetPath().pathString)
    if not sensors or not filters:
        raise RuntimeError(f"no robot links ({len(sensors)}) or fixture bodies ({len(filters)}) found")
    view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
        sensors, filter_patterns=[filters] * len(sensors), max_contact_data_count=0)
    if view.sensor_count != len(sensors) or view.filter_count != len(filters):
        raise RuntimeError(f"contact view has {view.sensor_count} x {view.filter_count}, "
                           f"expected {len(sensors)} x {len(filters)}")
    dt = float(getattr(base, "physics_dt", None) or base.sim.get_physics_dt())
    print(f"[slidekit] contact view for {spec.name}: {len(sensors)} robot links x parts {parts}", flush=True)
    return view, links, parts, dt


def _robot_contacts(base, spec: Fixture) -> dict | None:
    """For each part of the fixture, the robot links touching it at the last physics step, as "arm/link": from PhysX's
    contact force matrix between every link of both arms and the fixture's rigid bodies. None if PhysX can't report it."""
    v = _CONTACT.get(spec.name)
    if v is None:
        try:
            v = _CONTACT[spec.name] = _contact_view(base, spec)
        except Exception as e:  # noqa: BLE001
            v = _CONTACT[spec.name] = f"{type(e).__name__}: {e}"
            print(f"[slidekit] contact view for {spec.name} failed: {v}", flush=True)
    if isinstance(v, str):
        return None
    view, links, parts, dt = v
    n = np.linalg.norm(_host(view.get_contact_force_matrix(dt=dt)).reshape(len(links), len(parts), 3), axis=-1)
    return {p: [f"{a}/{link}" for (a, link), f in zip(links, n[:, j]) if f > CONTACT_N] for j, p in enumerate(parts)}


def _patch(scene) -> None:
    if getattr(scene, "_slidekit", False):
        return
    plain_object, plain_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Fixture):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return plain_object(spec)

    def object_states(env, names):
        plain = [n for n in names if n not in _SPECS]
        out = plain_states(env, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                out[n] = sys.modules[__name__]._state(env, _SPECS[n])
        return {n: out[n] for n in names}

    scene.task_object, scene.object_states, scene._slidekit = task_object, object_states, True


class _Hook(importlib.abc.MetaPathFinder):
    """Patches yam_arena.scene right after it is first imported."""

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
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if "yam_arena.scene" in sys.modules:
    _patch(sys.modules["yam_arena.scene"])
elif not any(isinstance(f, _Hook) for f in sys.meta_path):
    sys.meta_path.insert(0, _Hook())
