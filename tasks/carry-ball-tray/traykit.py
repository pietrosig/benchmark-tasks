"""traykit: one loose rigid object built from several boxes, for yam_arena task files.

yam_arena's own ``Box`` is a single primitive, so it cannot make a tray with a rim. A ``Compound`` is one rigid body
with a box collider per part. Importing this module hooks ``yam_arena.scene`` when it is imported, without modifying
yam_arena. The scorer's input needs no change: a ``Compound`` is an ordinary rigid object in the scene.

Pure Python at import time; everything that needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from dataclasses import dataclass

import numpy as np
from yam_arena.task_file import Box

__all__ = ["Compound", "compound", "install"]

Color = tuple[float, float, float]
BoxSpec = tuple[tuple[float, float, float], tuple[float, float, float], Color]  # centre, full size (m), sRGB colour


@dataclass(frozen=True)
class Compound(Box):
    """A loose rigid body of ``boxes``, given relative to ``pos``. It subclasses ``Box`` so yam_arena's task-file
    checks accept it; ``pos`` is the centre of the bounding box and ``size`` the bounding box's size."""

    boxes: tuple[BoxSpec, ...] = ()
    friction: float = 1.0

    def _check(self) -> None:
        super()._check()
        if self.mass is None:
            raise ValueError(f"{self.name}: a Compound is loose; give it a mass")
        if not self.boxes:
            raise ValueError(f"{self.name}: needs at least one box")


def compound(name, pos, boxes, mass, color=(0.7, 0.7, 0.7), friction=1.0) -> Compound:
    """Build a Compound from ``boxes`` given in any frame: they are recentred on their bounding box, which is put at
    ``pos``."""
    lo = np.min([np.array(c) - np.array(s) / 2 for c, s, _ in boxes], axis=0)
    hi = np.max([np.array(c) + np.array(s) / 2 for c, s, _ in boxes], axis=0)
    mid = (lo + hi) / 2
    centred = tuple((tuple((np.array(c) - mid).tolist()), tuple(s), k) for c, s, k in boxes)
    return Compound(name=name, pos=tuple(pos), size=tuple((hi - lo).tolist()), color=color, mass=mass, boxes=centred,
                    friction=friction)


_SPECS: dict[str, Compound] = {}


def _spawn_compound(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the body under ``prim_path`` with pxr: the rigid body at the root, a box collider per part below it."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec = _SPECS[cfg.compound_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    qx, qy, qz, qw = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(qw, qx, qy, qz))
    prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(spec.mass)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    body.CreateMaxDepenetrationVelocityAttr(1.0)
    body.CreateSolverPositionIterationCountAttr(32)

    physics_material = f"{prim_path}/Looks/physics"
    material = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    material.func(physics_material, material)
    looks: dict[Color, str] = {}

    def look(color: Color) -> str:
        if color not in looks:
            path = f"{prim_path}/Looks/colour_{len(looks)}"
            surface = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.4)
            surface.func(path, surface)
            looks[color] = path
        return looks[color]

    for i, (centre, size, color) in enumerate(spec.boxes):
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/box_{i}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        part = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(part)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(part)
        collision.CreateContactOffsetAttr(0.002)
        collision.CreateRestOffsetAttr(0.0)
        bind_physics_material(str(part.GetPath()), physics_material, stage=stage)
        bind_visual_material(str(part.GetPath()), look(color), stage=stage)
    return prim


def _arena_object(spec: Compound):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class CompoundSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_compound)
        compound_name: str = ""

    return Object(
        name=spec.name,
        object_type=ObjectType.RIGID,
        spawner_cfg=CompoundSpawnerCfg(compound_name=spec.name),
        initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    )


def _patch_scene(scene) -> None:
    if getattr(scene, "_traykit_patched", False):
        return
    base_task_object = scene.task_object

    def task_object(spec):
        if isinstance(spec, Compound):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return base_task_object(spec)

    scene.task_object = task_object
    scene._traykit_patched = True


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
