"""Offline preview of the task's layout and scripted demo, without Isaac Sim or a GPU.

    python preview.py ~/preview.png              # write the picture there (keep it out of the task folder)

It loads task.py the way the embodiment does (yam_arena.task_file.load), draws the tray, the balls and the fixture in
a top, side and front view, the arm bases and the reach limits from the yam-isaac-lab-arena README, the demo's
tool-centre-point path for each arm, and each arm's links at a few moments of the demo (forward kinematics on the YAM
URDF). It also checks the demo with yam_arena.demo.ArmPlan: the worst IK error and the joint limits.

It needs numpy, matplotlib and a checkout of yam-isaac-lab-arena on the path (set YAM_ARENA to its folder, default
~/yam-isaac-lab-arena). It shows geometry and reach only: no physics, collisions, textures or camera views.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, os.path.expanduser(os.environ.get("YAM_ARENA", "~/yam-isaac-lab-arena")))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from yam_arena import YAM_URDF, ik, station  # noqa: E402
from yam_arena.constants import ARM_JOINTS, CONTROL_HZ  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, Sphere, load  # noqa: E402
from yam_arena.urdf import forward_kinematics  # noqa: E402

sys.path.insert(0, str(HERE))
from traykit import Compound  # noqa: E402

VIEWS = (("top view (x forward, y left)", 0, 1), ("side view from the right (x, z)", 0, 2),
         ("front view from the arms (y, z)", 1, 2))
ARM_COLOURS = {"left": "tab:green", "right": "tab:purple"}
LINKS = ["base", "link1", "link2", "link3", "link4", "link5", "gripper"]


def world_boxes(task):
    """(centre, size, colour) in the world for every box in the scene."""
    out = []
    for obj in task.objects:
        if isinstance(obj, Compound):
            out += [(np.array(obj.pos) + c, np.array(s), k) for c, s, k in obj.boxes]
        elif isinstance(obj, Box):
            out.append((np.array(obj.pos), np.array(obj.size), obj.color))
    return out


def arm_links(q, base) -> np.ndarray:
    poses = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q)})
    points = [poses[name][:3, 3] for name in LINKS]
    tcp, _ = ik.tcp_pose(np.asarray(q), (0.0, 0.0, 0.0))
    return np.array(points + [tcp]) + np.asarray(base)


def check_plans(task):
    plans, report, frames = {}, [], {}
    for arm, waypoints in task.demo().items():
        plan = ArmPlan(list(waypoints), station.BASES[arm])
        steps = round(plan.duration * CONTROL_HZ) + 5
        actions, path = [], []
        for k in range(steps):
            t = (k + 1) / CONTROL_HZ
            action, _ = plan.action7(t)
            actions.append(action)
            path.append(plan.target(t)[0])
        actions = np.array(actions)
        inside = bool(np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER))
        report.append(f"{arm}: {plan.duration:.1f} s, worst IK error {plan.worst_ik_mm:.2f} mm / "
                      f"{plan.worst_ik_deg:.2f} deg, joints within limits: {inside}")
        plans[arm] = (plan, np.array(path), actions)
        ends = np.cumsum([w.duration for w in waypoints])
        frames[arm] = [(0.0, actions[0, :6])] + [(t, actions[min(len(actions) - 1, round(t * CONTROL_HZ) - 1), :6])
                                                 for t in ends[[2, 4, -1]]]
    return plans, report, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", nargs="?", default="preview.png", help="output image (default: ./preview.png)")
    args = parser.parse_args()

    task = load(HERE / "task.py")
    top = TABLE_TOP[task.scene]
    plans, report, frames = check_plans(task)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, (title, i, j) in zip(axes, VIEWS):
        for centre, size, colour in world_boxes(task):
            ax.add_patch(Rectangle((centre[i] - size[i] / 2, centre[j] - size[j] / 2), size[i], size[j],
                                   facecolor=colour, edgecolor="k", linewidth=0.4, alpha=0.55))
        for obj in task.objects:
            if isinstance(obj, Sphere):
                ax.add_patch(Circle((obj.pos[i], obj.pos[j]), obj.radius, facecolor="silver", edgecolor="k",
                                    linewidth=0.4))
        for arm, base in station.BASES.items():
            ax.plot(base[i], base[j], "s", color=ARM_COLOURS[arm], markersize=9)
            _, path, _ = plans.get(arm, (None, None, None))
            if path is not None:
                ax.plot(path[:, i], path[:, j], "-", color=ARM_COLOURS[arm], linewidth=1.2, label=f"{arm} TCP path")
            for n, (_, q) in enumerate(frames.get(arm, [])):
                pts = arm_links(q, base)
                ax.plot(pts[:, i], pts[:, j], "o-", color=ARM_COLOURS[arm], alpha=0.25 + 0.18 * n, linewidth=1,
                        markersize=2)
        if j == 2:
            ax.axhline(top, color="saddlebrown", linewidth=1)
        if (i, j) == (0, 1):  # reach limits at table height (README "Your own task"): 0.10-0.50 m ahead of each base
            for base in station.BASES.values():
                for x in (0.10, 0.50):
                    ax.plot([base[0] + x] * 2, [base[1] - 0.06, base[1] + 0.06], "k--", linewidth=0.6)
            ax.plot(0.40, 0.0, "kx")
            ax.annotate("midline reach 0.40 m", (0.40, 0.0), (0.42, 0.05), fontsize=7)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle(f"{HERE.name}\n" + "   |   ".join(report), fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("\n".join(report))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
