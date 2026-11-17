"""fruitkit: soft fruit for yam_arena task files.

A ``Fruit`` is a free rigid body shaped like an ellipsoid (a kiwi lying on its side), collided as one convex hull, whose
contacts are compliant: PhysX's compliant-contact spring (``stiffness`` N/m per contact, ``damping`` N s/m) lets a
squeezing finger sink into it in proportion to the force, the way a ripe fruit gives under the thumb and a hard one
hardly does. Importing this module hooks ``yam_arena.scene`` (without modifying yam_arena); the task's ``score()`` then
gets a ``FruitState`` for each fruit: the usual pose and velocities, plus the contact force each robot link puts on it
right now (from a PhysX contact view between every link of both arms and every fruit), and where each gripper is.

Pure Python at import time; everything that needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena.task_file import Box, ObjectState, Sphere

__all__ = ["Bowl", "Fruit", "FruitState", "fruit"]


@dataclass(frozen=True)
class Fruit(Sphere):
    """An ellipsoid fruit, its semi-axes ``semi`` (m) along x, y and z before ``yaw``; ``radius`` is the largest."""

    semi: tuple = (0.034, 0.026, 0.026)
    stiffness: float = 1e5  # compliant-contact spring, N/m, per contact
    damping: float = 50.0  # N s/m
    friction: float = 0.9
    spot_color: tuple = (0.30, 0.22, 0.13)  # the dried flower end and the stem end


@dataclass(frozen=True)
class FruitState(ObjectState):
    force: dict = field(default_factory=dict)  # "arm/link" -> contact force (N) that robot link puts on the fruit now
    hands: dict = field(default_factory=dict)  # arm -> {"tcp": world xyz between the fingertips, "gap": pad gap (m)}

    @property
    def robot_force(self) -> float:
        """The largest contact force any robot link puts on the fruit right now (N); nan if PhysX can't tell."""
        return max(self.force.values(), default=0.0) if self.force is not None else float("nan")


def fruit(name, pos, semi, mass, stiffness, damping, color, yaw=0.0, friction=0.9) -> Fruit:
    return Fruit(name=name, pos=tuple(pos), radius=float(max(semi)), mass=mass, color=tuple(color), yaw=yaw,
                 semi=tuple(semi), stiffness=float(stiffness), damping=float(damping), friction=friction)


@dataclass(frozen=True)
class Bowl(Box):
    """A round bowl fixed to the table: ``pos`` is the centre of its base on the table top. Its wall flares straight
    from ``base_r`` at the bottom to ``rim_r`` at height ``depth``; it is ``wall`` thick. Drawn as a smooth surface
    of revolution, collided as a flat disc and a ring of ``segments`` thin slanted plates."""

    base_r: float = 0.045
    rim_r: float = 0.095
    depth: float = 0.065
    wall: float = 0.005
    segments: int = 32


def bowl(name, pos, base_r, rim_r, depth, color, wall=0.005) -> Bowl:
    return Bowl(name=name, pos=tuple(pos), size=(2 * rim_r, 2 * rim_r, depth), color=tuple(color), base_r=base_r,
                rim_r=rim_r, depth=depth, wall=wall)


_SPECS: dict = {}
_BOWLS: dict = {}
_CONTACT: dict = {}  # "view" -> (contact view, robot links, fruit names, dt), or the error that stopped it
CONTACT_N = 0.05  # N: a contact force above this between a robot link and a fruit counts as touching


def _ellipsoid(semi, n_lat=14, n_lon=24):
    """Vertices, face counts and indices of a UV ellipsoid."""
    a, b, c = semi
    pts = [(0.0, 0.0, c)]
    for i in range(1, n_lat):
        th = math.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * math.pi * j / n_lon
            pts.append((a * math.sin(th) * math.cos(ph), b * math.sin(th) * math.sin(ph), c * math.cos(th)))
    pts.append((0.0, 0.0, -c))
    counts, idx = [], []
    for j in range(n_lon):
        counts.append(3)
        idx += [0, 1 + j, 1 + (j + 1) % n_lon]
    for i in range(n_lat - 2):
        r0, r1 = 1 + i * n_lon, 1 + (i + 1) * n_lon
        for j in range(n_lon):
            k = (j + 1) % n_lon
            counts.append(4)
            idx += [r0 + j, r1 + j, r1 + k, r0 + k]
    last, bottom = 1 + (n_lat - 2) * n_lon, len(pts) - 1
    for j in range(n_lon):
        counts.append(3)
        idx += [last + j, bottom, last + (j + 1) % n_lon]
    return pts, counts, idx


def _spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt
    from yam_arena.station import linear_rgb

    spec = _SPECS[cfg.fruit_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0, 0, 0))))
    qx, qy, qz, qw = orientation if orientation is not None else (0, 0, 0, 1)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(qw, qx, qy, qz))
    p = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(p)
    UsdPhysics.MassAPI.Apply(p).CreateMassAttr(spec.mass)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
    body.CreateMaxDepenetrationVelocityAttr(0.5)
    body.CreateSolverPositionIterationCountAttr(32)
    body.CreateSolverVelocityIterationCountAttr(4)
    body.CreateAngularDampingAttr(0.5)

    mat = f"{prim_path}/Looks/physics"
    UsdShade.Material.Define(stage, mat)
    mp = stage.GetPrimAtPath(mat)
    m = UsdPhysics.MaterialAPI.Apply(mp)
    m.CreateStaticFrictionAttr(spec.friction)
    m.CreateDynamicFrictionAttr(spec.friction)
    m.CreateRestitutionAttr(0.0)
    px = PhysxSchema.PhysxMaterialAPI.Apply(mp)
    px.CreateCompliantContactStiffnessAttr(spec.stiffness)
    px.CreateCompliantContactDampingAttr(spec.damping)

    pts, counts, idx = _ellipsoid(spec.semi)
    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/skin")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*q) for q in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateSubdivisionSchemeAttr("none")
    g = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(g)
    UsdPhysics.MeshCollisionAPI.Apply(g).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    col = PhysxSchema.PhysxCollisionAPI.Apply(g)
    col.CreateContactOffsetAttr(0.004)
    col.CreateRestOffsetAttr(0.0)
    bind_physics_material(str(g.GetPath()), mat, stage=stage)
    skin = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.95)
    skin.func(f"{prim_path}/Looks/skin", skin)
    bind_visual_material(str(g.GetPath()), f"{prim_path}/Looks/skin", stage=stage)
    # the two ends: small dark caps (look only)
    spot = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.spot_color), roughness=0.9)
    spot.func(f"{prim_path}/Looks/spot", spot)
    a = spec.semi[0]
    for k, sx in enumerate((1, -1)):
        s = UsdGeom.Sphere.Define(stage, f"{prim_path}/end_{k}")
        s.CreateRadiusAttr(0.006 if k == 0 else 0.004)
        s.AddTranslateOp().Set(Gf.Vec3d(sx * (a - 0.0035), 0, 0))
        s.AddScaleOp().Set(Gf.Vec3f(0.5, 1.0, 1.0))
        bind_visual_material(str(s.GetPrim().GetPath()), f"{prim_path}/Looks/spot", stage=stage)
    return p


def _arena_object(spec: Fruit):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class FruitCfg(SpawnerCfg):
        func: object = clone(_spawn)
        fruit_name: str = ""

    yaw = spec.yaw
    return Object(name=spec.name, object_type=ObjectType.RIGID, spawner_cfg=FruitCfg(fruit_name=spec.name),
                  initial_pose=Pose(position_xyz=tuple(spec.pos),
                                    rotation_xyzw=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))))


def _spawn_bowl(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, Vt
    from yam_arena.station import linear_rgb

    spec = _BOWLS[cfg.bowl_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0, 0, 0))))
    mat = f"{prim_path}/Looks/physics"
    m = sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.4)
    m.func(mat, m)
    glaze = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.25)
    glaze.func(f"{prim_path}/Looks/glaze", glaze)
    b, r, h, t = spec.base_r, spec.rim_r, spec.depth, spec.wall
    # the look: a surface of revolution through this profile (outside up, over the rim, inside down, the floor)
    prof = [(0.0, 0.0), (b, 0.0), (r, h), (r - t, h), (b - t * 0.8, t), (0.0, t)]
    n = 48
    pts = [(pr * math.cos(2 * math.pi * k / n), pr * math.sin(2 * math.pi * k / n), pz) for pr, pz in prof
           for k in range(n)]
    counts, idx = [], []
    for i in range(len(prof) - 1):
        for k in range(n):
            k1 = (k + 1) % n
            counts.append(4)
            idx += [i * n + k, i * n + k1, (i + 1) * n + k1, (i + 1) * n + k]
    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/look")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*q) for q in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    bind_visual_material(str(mesh.GetPrim().GetPath()), f"{prim_path}/Looks/glaze", stage=stage)

    def collider(g):
        UsdPhysics.CollisionAPI.Apply(g)
        col = PhysxSchema.PhysxCollisionAPI.Apply(g)
        col.CreateContactOffsetAttr(0.003)
        col.CreateRestOffsetAttr(0.0)
        bind_physics_material(str(g.GetPath()), mat, stage=stage)
        UsdGeom.Imageable(g).CreateVisibilityAttr("invisible")

    floor = UsdGeom.Cylinder.Define(stage, f"{prim_path}/floor")
    floor.CreateRadiusAttr(b)
    floor.CreateHeightAttr(t)
    floor.CreateAxisAttr("Z")
    floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, t / 2))
    collider(floor.GetPrim())
    slant = math.atan2(h, r - b)  # the wall's angle above the horizontal
    length = math.hypot(h, r - b)
    rm, zm = (b + r) / 2 - t / 2 * math.sin(slant), h / 2 + t / 2 * math.cos(slant)
    width = 2 * math.pi * (r + 0.004) / spec.segments * 1.08
    for k in range(spec.segments):
        a = 2 * math.pi * k / spec.segments
        plate = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        plate.CreateSizeAttr(1.0)
        plate.AddTranslateOp().Set(Gf.Vec3d(rm * math.cos(a), rm * math.sin(a), zm))
        # turn about z by a, then tilt the plate (its x along the profile) up by the slant
        qz = Gf.Quatd(math.cos(a / 2), 0, 0, math.sin(a / 2))
        qy = Gf.Quatd(math.cos(-slant / 2), 0, math.sin(-slant / 2), 0)
        plate.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(qz * qy)
        plate.AddScaleOp().Set(Gf.Vec3f(length, width * (b + r) / 2 / (r + 0.004) * 1.3, t))
        collider(plate.GetPrim())
    return root.GetPrim()


def _arena_bowl(spec: Bowl):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class BowlCfg(SpawnerCfg):
        func: object = clone(_spawn_bowl)
        bowl_name: str = ""

    return Object(name=spec.name, object_type=ObjectType.BASE, spawner_cfg=BowlCfg(bowl_name=spec.name),
                  initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


def _np(x):
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _host(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    import warp as wp
    return wp.to_torch(x).cpu().numpy()


def _hands(base) -> dict:
    """Where each gripper's tool point is (world, like the task's positions) and the gap between its pads (m)."""
    from yam_arena.constants import FINGER_JOINTS
    from yam_arena.runtime import tcp_world

    origin = _np(base.scene.env_origins)[0]
    out = {}
    for arm in ("left_robot", "right_robot"):
        if arm not in base.scene.keys():
            continue
        robot = base.scene[arm]
        fingers = robot.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        gap = float(-_np(robot.data.joint_pos)[0][fingers].sum())
        out[arm] = {"tcp": tcp_world(robot) - origin, "gap": gap}
    return out


def _rigid_bodies(stage, root: str) -> list:
    from pxr import Usd, UsdPhysics
    return [p for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.RigidBodyAPI)]


def _contact_view(base):
    """A PhysX contact view from every link of both arms (the YAM spawns them with contact reporting on) to each fruit."""
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
    names, filters = [], []
    for n in _SPECS:
        if n in base.scene.keys():
            bodies = _rigid_bodies(stage, root(n))
            if bodies:
                names.append(n)
                filters.append(bodies[0].GetPath().pathString)
    if not sensors or not filters:
        raise RuntimeError(f"no robot links ({len(sensors)}) or fruit bodies ({len(filters)}) found")
    view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
        sensors, filter_patterns=[filters] * len(sensors), max_contact_data_count=0)
    if view.sensor_count != len(sensors) or view.filter_count != len(filters):
        raise RuntimeError(f"contact view has {view.sensor_count} x {view.filter_count}, "
                           f"expected {len(sensors)} x {len(filters)}")
    dt = float(getattr(base, "physics_dt", None) or base.sim.get_physics_dt())
    print(f"[fruitkit] contact view: {len(sensors)} robot links x fruits {names}", flush=True)
    return view, links, names, dt


def _forces(base) -> dict | None:
    """fruit name -> {"arm/link": N} for every robot link touching it at the last physics step; None if unknown."""
    v = _CONTACT.get("view")
    if v is None:
        try:
            v = _CONTACT["view"] = _contact_view(base)
        except Exception as e:  # noqa: BLE001
            v = _CONTACT["view"] = f"{type(e).__name__}: {e}"
            print(f"[fruitkit] contact view failed: {v}", flush=True)
    if isinstance(v, str):
        return None
    view, links, names, dt = v
    n = np.linalg.norm(_host(view.get_contact_force_matrix(dt=dt)).reshape(len(links), len(names), 3), axis=-1)
    return {f: {f"{a}/{link}": round(float(x), 2) for (a, link), x in zip(links, n[:, j]) if x > CONTACT_N}
            for j, f in enumerate(names)}


def _patch(scene) -> None:
    if getattr(scene, "_fruitkit", False):
        return
    plain_object, plain_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Fruit):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        if isinstance(spec, Bowl):
            _BOWLS[spec.name] = spec
            return _arena_bowl(spec)
        return plain_object(spec)

    def object_states(env, names):
        out = plain_states(env, names)
        fruits = [n for n in names if n in _SPECS]
        if fruits:
            base = env.unwrapped
            forces, hands = _forces(base), _hands(base)
            for n in fruits:
                s = out[n]
                out[n] = FruitState(pos=s.pos, quat_xyzw=s.quat_xyzw, lin_vel=s.lin_vel, ang_vel=s.ang_vel, low=s.low,
                                    high=s.high, force=None if forces is None else forces.get(n, {}), hands=hands)
        return out

    scene.task_object, scene.object_states, scene._fruitkit = task_object, object_states, True


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
