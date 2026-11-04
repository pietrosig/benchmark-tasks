"""drawerkit: fixed furniture with sliding drawers, for yam_arena task files.

A task file uses it like this (keep this file next to the task file):

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from drawerkit import Drawer, cabinet

    CABINET = cabinet("cabinet", pos=(0.30, 0.0, TOP), boxes=[...], drawers=[Drawer("top_drawer", ...), ...])
    OBJECTS = [CABINET, ...]

yam_arena's own task-file objects are single loose or fixed primitives, with no joints. A ``Cabinet`` is one fixed
body made of several box colliders, plus ``drawers``: each drawer is its own body of boxes on a prismatic joint along
the cabinet's x axis, opening towards -x by up to ``travel`` metres, with viscous damping on the slide. The cabinet
is anchored to the world, so only its drawers move.

Importing this module hooks ``yam_arena.scene`` when it is imported, without modifying yam_arena: a ``Cabinet`` in
``OBJECTS`` is spawned as a fixed-base PhysX articulation, and ``success()/score()/distance()`` receive a
:class:`CabinetState` for it (an ``ObjectState`` plus each drawer's opening and opening speed).

Pure Python at import time; everything that needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena.task_file import Box, ObjectState

__all__ = ["Cabinet", "CabinetState", "Drawer", "cabinet", "closed_bounds", "install"]

Color = tuple[float, float, float]
BoxSpec = tuple[tuple[float, float, float], tuple[float, float, float], Color]  # centre, full size (m), sRGB colour


# -- specification ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Drawer:
    """A drawer on a slide along the cabinet's x axis.

    ``closed`` is the drawer frame's origin in the cabinet frame when the drawer is shut; ``boxes`` are in the drawer
    frame. The drawer opens towards -x by up to ``travel`` m; ``damping`` (N per m/s) resists slide motion."""

    name: str
    closed: tuple[float, float, float]
    boxes: tuple[BoxSpec, ...]
    travel: float = 0.20
    damping: float = 5.0
    mass: float = 0.4


@dataclass(frozen=True)
class Cabinet(Box):
    """Fixed furniture: a body of ``boxes`` (cabinet frame, metres) and sliding ``drawers``.

    It subclasses ``Box`` only so yam_arena's task-file checks accept it. ``pos`` is the cabinet frame's origin in the
    world, ``size`` is the closed cabinet's bounding box (centred at ``bbox_centre`` in the cabinet frame), and it has
    no ``mass``: the cabinet is anchored and never moves."""

    boxes: tuple[BoxSpec, ...] = ()
    drawers: tuple[Drawer, ...] = ()
    bbox_centre: tuple[float, float, float] = (0.0, 0.0, 0.0)
    friction: float = 0.8

    def _check(self) -> None:
        super()._check()
        if self.mass is not None:
            raise ValueError(f"{self.name}: a Cabinet is fixed; leave out its mass")
        if not self.boxes:
            raise ValueError(f"{self.name}: a Cabinet needs at least one box")
        names = [d.name for d in self.drawers]
        if len(set(names)) != len(names) or not all(n.isidentifier() for n in names):
            raise ValueError(f"{self.name}: drawer names must be unique Python identifiers, got {names}")
        for d in self.drawers:
            if not (d.travel > 0 and d.damping >= 0 and d.mass > 0 and d.boxes):
                raise ValueError(f"{self.name}.{d.name}: needs boxes, travel > 0, damping >= 0 and mass > 0")


def _extent(boxes) -> tuple[np.ndarray, np.ndarray]:
    lo = np.min([np.array(c) - np.array(s) / 2 for c, s, _ in boxes], axis=0)
    hi = np.max([np.array(c) + np.array(s) / 2 for c, s, _ in boxes], axis=0)
    return lo, hi


def cabinet(name, pos, boxes, drawers=(), yaw=0.0, color=(0.7, 0.7, 0.7), friction=0.8) -> Cabinet:
    """Build a Cabinet; its closed bounding box is computed from the body and the shut drawers."""
    shapes = list(boxes) + [((np.array(d.closed) + c).tolist(), s, k) for d in drawers for c, s, k in d.boxes]
    lo, hi = _extent(shapes)
    return Cabinet(name=name, pos=tuple(pos), size=tuple((hi - lo).tolist()), color=color, yaw=yaw,
                   boxes=tuple(boxes), drawers=tuple(drawers), bbox_centre=tuple(((lo + hi) / 2).tolist()),
                   friction=friction)


@dataclass(frozen=True)
class CabinetState(ObjectState):
    """The cabinet frame's origin (``pos``) and orientation, its closed bounding box, and per drawer:
    ``opening`` (m, 0 when shut, positive when pulled out) and ``speed`` (m/s, positive while opening)."""

    opening: dict = field(default_factory=dict)
    speed: dict = field(default_factory=dict)


# -- spawning (Isaac Sim, imported lazily) --------------------------------------------------------------------------

_SPECS: dict[str, Cabinet] = {}


def _joint_name(drawer: Drawer) -> str:
    return f"{drawer.name}_slide"


def _spawn_cabinet(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the cabinet under ``prim_path`` with pxr: a body, one body per drawer, a prismatic joint per drawer,
    and a fixed joint anchoring the body to the world."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec = _SPECS[cfg.cabinet_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    qx, qy, qz, qw = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(qw, qx, qy, qz))

    physics_material = f"{prim_path}/Looks/physics"
    material = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    material.func(physics_material, material)
    looks: dict[Color, str] = {}

    def look(color: Color) -> str:
        if color not in looks:
            path = f"{prim_path}/Looks/colour_{len(looks)}"
            surface = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.5)
            surface.func(path, surface)
            looks[color] = path
        return looks[color]

    def rigid_body(path: str, pos, mass: float):
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        body.CreateMaxDepenetrationVelocityAttr(1.0)
        body.CreateSolverPositionIterationCountAttr(32)
        return prim

    def add_boxes(path: str, boxes):
        for i, (centre, size, color) in enumerate(boxes):
            cube = UsdGeom.Cube.Define(stage, f"{path}/box_{i}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            prim = cube.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(0.001)
            collision.CreateRestOffsetAttr(0.0)
            bind_physics_material(str(prim.GetPath()), physics_material, stage=stage)
            bind_visual_material(str(prim.GetPath()), look(color), stage=stage)

    body_path = f"{prim_path}/body"
    rigid_body(body_path, (0.0, 0.0, 0.0), 5.0)  # anchored below, so its mass only matters to the solver
    add_boxes(body_path, spec.boxes)

    for drawer in spec.drawers:
        path = f"{prim_path}/{drawer.name}"
        rigid_body(path, drawer.closed, drawer.mass)
        add_boxes(path, drawer.boxes)
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/{_joint_name(drawer)}")
        joint.CreateAxisAttr("X")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*drawer.closed))
        joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        joint.CreateLowerLimitAttr(-drawer.travel)  # joint position is negative while the drawer is out
        joint.CreateUpperLimitAttr(0.0)

    body_prim = stage.GetPrimAtPath(body_path)
    UsdPhysics.ArticulationRootAPI.Apply(body_prim)
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(body_prim)
    articulation.CreateEnabledSelfCollisionsAttr(False)  # drawers are held by their slides, not by the carcass
    articulation.CreateSolverPositionIterationCountAttr(32)
    articulation.CreateSolverVelocityIterationCountAttr(4)

    anchor = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/anchor")  # body0 left empty: the world
    anchor.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
    world = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(0)
    where, rotation = world.ExtractTranslation(), world.ExtractRotationQuat()
    anchor.CreateLocalPos0Attr(Gf.Vec3f(*where))
    anchor.CreateLocalRot0Attr(Gf.Quatf(rotation.GetReal(), *rotation.GetImaginary()))
    anchor.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    anchor.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    return root.GetPrim()


def _arena_object(spec: Cabinet):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class CabinetSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_cabinet)
        cabinet_name: str = ""

    class CabinetObject(Object):
        def _generate_articulation_cfg(self):
            # Unpowered slides: no stiffness, only the drawer's viscous damping.
            actuators = {d.name: ImplicitActuatorCfg(joint_names_expr=[_joint_name(d)], stiffness=0.0,
                                                     damping=d.damping) for d in spec.drawers}
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators)
            return self._add_initial_pose_to_cfg(cfg)

    return CabinetObject(
        name=spec.name,
        object_type=ObjectType.ARTICULATION,
        spawner_cfg=CabinetSpawnerCfg(cabinet_name=spec.name),
        initial_pose=Pose(position_xyz=tuple(spec.pos),
                          rotation_xyzw=(0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))),
    )


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def closed_bounds(spec: Cabinet) -> tuple[np.ndarray, np.ndarray]:
    """World-aligned bounding box of the shut cabinet in the world (env) frame."""
    c, s = math.cos(spec.yaw), math.sin(spec.yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    corners = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)]) * np.array(spec.size) / 2
    points = np.array(spec.pos) + (rot @ (corners + np.array(spec.bbox_centre)).T).T
    return points.min(axis=0), points.max(axis=0)


def _cabinet_state(environment, spec: Cabinet) -> CabinetState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    names = list(asset.joint_names)
    q, qd = _np(asset.data.joint_pos)[0], _np(asset.data.joint_vel)[0]
    opening = {d.name: float(-q[names.index(_joint_name(d))]) for d in spec.drawers}
    speed = {d.name: float(-qd[names.index(_joint_name(d))]) for d in spec.drawers}
    low, high = closed_bounds(spec)
    return CabinetState(pos=pose[:3] - origin, quat_xyzw=pose[3:7], lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                        low=low, high=high, opening=opening, speed=speed)


# -- hooking yam_arena ----------------------------------------------------------------------------------------------


def _patch_scene(scene) -> None:
    if getattr(scene, "_drawerkit_patched", False):
        return
    base_task_object, base_object_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Cabinet):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return base_task_object(spec)

    def object_states(environment, names):
        plain = [n for n in names if n not in _SPECS]
        states = base_object_states(environment, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                states[n] = _cabinet_state(environment, _SPECS[n])
        return {n: states[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._drawerkit_patched = True


class _SceneHook(importlib.abc.MetaPathFinder):
    """Patches ``yam_arena.scene`` right after it is first imported."""

    target = "yam_arena.scene"

    def find_spec(self, fullname, path, target=None):
        if fullname != self.target:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch_scene(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    """Hook yam_arena.scene: now if it is already imported, otherwise when it is. Safe to call more than once."""
    if _SceneHook.target in sys.modules:
        _patch_scene(sys.modules[_SceneHook.target])
    if not any(isinstance(f, _SceneHook) for f in sys.meta_path):
        sys.meta_path.insert(0, _SceneHook())


install()
