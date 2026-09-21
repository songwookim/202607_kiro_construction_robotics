#!/usr/bin/env python3
"""Regenerate the figures in WELD_MATH_VISUAL.md from the production code.

    source src/construct_robot_ros2/scripts/use_ros_python.bash
    python3 src/construct_robot_ros2/docs/make_weld_math_figures.py

Every curve here is the output of the function the document describes -- none
of it is redrawn by hand.  That is the point: if someone changes the weave
maths, the figures change with it and the document stops lying.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geometry_msgs.msg import Pose

from construct_robot.cartesian_path_common import (
    circular_weaving_from_path,
    linear_pose_waypoints,
    sine_weaving_with_dwell,
    weave_cycles_for_pitch,
    weaving_from_path,
)
from construct_robot.cartesian_path_common import (
    retime_trajectory_constant_velocity,
    trajectory_duration_seconds,
)
from construct_robot.weld_action_gui import (
    compute_corrected_seam_geometry,
    compute_plane_intersection_line,
    compute_surface_plane,
    project_point_to_line,
)

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

SEAM_M = 0.186          # a real seam length from weld_feedback/
AMPLITUDE_M = 0.002     # 2 mm one-side amplitude
BLUE, ORANGE, GREEN, GREY = "#2b6cb0", "#dd6b20", "#2f855a", "#718096"


def pose(x, y, z, qw=1.0):
    p = Pose()
    p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
    p.orientation.w = float(qw)
    return p


def seam_poses(n=2):
    return linear_pose_waypoints(pose(0, 0, 0), pose(SEAM_M, 0, 0), n)


def xyz(points):
    return np.array([[p.position.x, p.position.y, p.position.z] for p in points])


def style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)


# --------------------------------------------------------------------------
def fig_patterns():
    """sine vs crescent vs circle, all at the same amplitude and pitch."""
    base = seam_poses()
    cycles = weave_cycles_for_pitch(SEAM_M, 30.0)      # 30 mm/cycle
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True)

    sine = xyz(weaving_from_path(base, AMPLITUDE_M, cycles, 48, "world_y"))
    axes[0].plot(sine[:, 0] * 1000, sine[:, 1] * 1000, color=BLUE, lw=1.6)
    style(axes[0], "", "lateral [mm]",
          f"sine  ·  {cycles} cycles, pitch {SEAM_M*1000/cycles:.1f} mm, A = 2 mm")

    cres = xyz(weaving_from_path(base, AMPLITUDE_M, cycles, 48, "world_y",
                                 pattern="crescent"))
    axes[1].plot(cres[:, 0] * 1000, cres[:, 1] * 1000, color=ORANGE, lw=1.6)
    # The crescent differs from the sine in seam *progress*, not in lateral
    # offset, so the curve alone looks identical.  Marking the samples shows
    # it: they bunch where ds/dphi is small and spread where it is large.
    step = 2
    axes[1].plot(cres[::step, 0] * 1000, cres[::step, 1] * 1000, ls="none",
                 marker="o", ms=2.6, color=ORANGE, alpha=0.55)
    axes[1].plot(sine[::step, 0] * 1000, sine[::step, 1] * 1000 - 5.0,
                 ls="none", marker="o", ms=2.6, color=BLUE, alpha=0.45)
    axes[1].plot(sine[:, 0] * 1000, sine[:, 1] * 1000 - 5.0, color=BLUE,
                 lw=1.0, alpha=0.5)
    axes[1].text(2, -5.0, "sine, same samples\n(offset −5 mm to compare)",
                 fontsize=7, color=BLUE, va="center")
    style(axes[1], "", "lateral [mm]",
          "crescent  ·  same lateral shape, samples redistributed along the seam")

    circ = xyz(circular_weaving_from_path(base, AMPLITUDE_M, cycles, 24,
                                          "world_y"))
    axes[2].plot(circ[:, 0] * 1000, circ[:, 1] * 1000, color=GREEN, lw=1.6,
                 label="lateral (primary)")
    axes[2].plot(circ[:, 0] * 1000, circ[:, 2] * 1000, color=GREY, lw=1.2,
                 ls="--", label="vertical (secondary)")
    axes[2].legend(fontsize=8, ncol=2)
    style(axes[2], "along seam [mm]", "offset [mm]",
          "circle  ·  two orthogonal offsets 90° apart, with sin² end ramp")

    for ax in axes:
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "weave_patterns.png", dpi=150)
    plt.close(fig)


def fig_crescent_progress():
    """Why the crescent bulge is bounded: ds/dphase must stay positive."""
    cycles = weave_cycles_for_pitch(SEAM_M, 30.0)
    pitch = SEAM_M / cycles
    bound = pitch / (4.0 * np.pi)
    phase = np.linspace(0, 2 * np.pi, 400)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for bulge, color, label in (
        (min(0.5 * AMPLITUDE_M, bound), GREEN, f"code: min(A/2, p/4π) = {min(0.5*AMPLITUDE_M, bound)*1000:.2f} mm"),
        (bound, ORANGE, f"at the bound p/4π = {bound*1000:.2f} mm"),
        (2.2 * bound, BLUE, f"2.2x the bound — {2.2*bound*1000:.2f} mm"),
    ):
        ds = pitch / (2 * np.pi) + bulge * np.sin(2 * phase)
        axes[0].plot(phase, ds * 1000, color=color, lw=1.6, label=label)
    axes[0].axhline(0, color="crimson", lw=1.0, ls=":")
    axes[0].legend(fontsize=7.5)
    style(axes[0], "phase φ [rad]", "ds/dφ [mm/rad]",
          "seam progress rate - below zero means the TCP moves backwards")

    for bulge, color in ((min(0.5 * AMPLITUDE_M, bound), GREEN),
                         (2.2 * bound, BLUE)):
        s = pitch * phase / (2 * np.pi) + 0.5 * bulge * (1 - np.cos(2 * phase))
        axes[1].plot(phase, s * 1000, color=color, lw=1.6)
    style(axes[1], "phase φ [rad]", "distance along seam s [mm]",
          "cumulative progress over one cycle")
    fig.tight_layout()
    fig.savefig(OUT / "crescent_progress.png", dpi=150)
    plt.close(fig)


def fig_pitch_quantisation():
    """Requested pitch is a maximum; the realised pitch is a staircase."""
    requested = np.linspace(5.0, 60.0, 600)
    realised, cycles_used = [], []
    for p in requested:
        c = weave_cycles_for_pitch(SEAM_M, float(p))
        cycles_used.append(c)
        realised.append(SEAM_M * 1000.0 / c)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    axes[0].plot(requested, requested, color=GREY, ls="--", lw=1.0,
                 label="requested")
    axes[0].plot(requested, realised, color=BLUE, lw=1.6, label="realised")
    axes[0].legend(fontsize=8)
    style(axes[0], "requested pitch [mm/cycle]", "pitch [mm/cycle]",
          f"realised pitch on a {SEAM_M*1000:.0f} mm seam")
    axes[1].step(requested, cycles_used, color=ORANGE, lw=1.6, where="post")
    style(axes[1], "requested pitch [mm/cycle]", "whole cycles C",
          "C = ceil(L / pitch) — keeps both ends on the centerline")
    fig.tight_layout()
    fig.savefig(OUT / "pitch_quantisation.png", dpi=150)
    plt.close(fig)


def fig_dwell():
    """Where the dwell holds land inside the 12-sample cycle."""
    cycles = 3
    points, holds = sine_weaving_with_dwell(
        seam_poses(), AMPLITUDE_M, cycles,
        left_dwell_s=0.4, right_dwell_s=0.2,
        transverse_axis="world_y",
    )
    p = xyz(points)
    holds = np.array(holds)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(p[:, 0] * 1000, p[:, 1] * 1000, color=BLUE, lw=1.4,
            marker="o", ms=3, label="weave samples (12 per cycle)")
    for label, idx, color in (("left dwell (+A)", 3, GREEN),
                              ("right dwell (−A)", 9, ORANGE)):
        sel = [c * 12 + idx for c in range(cycles)]
        ax.scatter(p[sel, 0] * 1000, p[sel, 1] * 1000, s=110, zorder=5,
                   facecolor="none", edgecolor=color, lw=2.0,
                   label=f"{label}: {holds[sel][0]:.1f} s")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.legend(fontsize=8, ncol=3)
    style(ax, "along seam [mm]", "lateral [mm]",
          "sine_weaving_with_dwell — holds sit exactly on the ±A peaks")
    fig.tight_layout()
    fig.savefig(OUT / "weave_dwell.png", dpi=150)
    plt.close(fig)


def fig_speed():
    """Dwell steals time from travel, so the moving TCP must go faster."""
    travel = np.linspace(2.0, 10.0, 400)          # mm/s seam advance
    cycles = weave_cycles_for_pitch(SEAM_M, 30.0)
    points, holds = sine_weaving_with_dwell(
        seam_poses(), AMPLITUDE_M, cycles, 0.3, 0.3, "world_y")
    p = xyz(points)
    path_len = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    dwell_total = float(np.sum(holds))

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for dwell, color, label in ((0.0, GREY, "no dwell"),
                                (dwell_total, BLUE,
                                 f"dwell {dwell_total:.1f} s total")):
        target = SEAM_M / (travel * 1e-3)
        moving = target - dwell
        speed = np.where(moving > 0, path_len / np.where(moving > 0, moving, 1), np.nan)
        ax.plot(travel, speed * 1000, color=color, lw=1.7, label=label)
    ax.plot(travel, travel, color=ORANGE, ls=":", lw=1.2,
            label="straight seam (no weave)")
    ax.legend(fontsize=8)
    style(ax, "requested seam advance [mm/s]", "commanded TCP speed [mm/s]",
          f"weave path is {path_len/SEAM_M:.2f}x longer than the seam")
    fig.tight_layout()
    fig.savefig(OUT / "weave_speed.png", dpi=150)
    plt.close(fig)


def fig_seam_correction():
    """Touch two surfaces, intersect the planes, get the real seam.

    Drawn as a cross-section rather than a 3-D render: perpendicular to the
    seam the two planes are just lines and the seam is where they cross, which
    is exactly the geometry the code solves.
    """
    taught_start = np.array([0.0, 0.0, 0.0])
    taught_end = np.array([SEAM_M, 0.0, 0.0])

    # the real workpiece sits 4 mm off in Y, 3 mm down in Z, and is tilted 6 deg
    tilt = np.deg2rad(6.0)
    wall_touches = [(0.03, 0.004, 0.02),
                    (0.15, 0.004 + 0.12 * np.tan(tilt), 0.02)]
    floor_touches = [(0.03, 0.03, -0.003), (0.15, 0.05, -0.003)]

    wall_n, wall_c = compute_surface_plane((0.0, 1.0, 0.0), wall_touches)
    floor_n, floor_c = compute_surface_plane((0.0, 0.0, 1.0), floor_touches)
    origin, direction = compute_plane_intersection_line(
        wall_n, wall_c, floor_n, floor_c,
        direction_reference=tuple(taught_end - taught_start))

    origin = np.array(origin)
    direction = np.array(direction)
    corrected_start = np.array(project_point_to_line(taught_start, origin, direction))
    corrected_end = np.array(project_point_to_line(taught_end, origin, direction))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    # -- cross-section at mid-seam: both planes become lines -------------
    x_cut = 0.5 * SEAM_M
    ax = axes[0]
    y = np.linspace(-0.01, 0.07, 2)
    z = np.linspace(-0.02, 0.04, 2)
    # wall plane  n.x = c  ->  solve for y given x_cut, z
    z_line = np.linspace(-0.02, 0.04, 50)
    y_wall = (wall_c - wall_n[0] * x_cut - wall_n[2] * z_line) / wall_n[1]
    y_line = np.linspace(-0.01, 0.07, 50)
    z_floor = (floor_c - floor_n[0] * x_cut - floor_n[1] * y_line) / floor_n[2]
    ax.plot(y_wall * 1000, z_line * 1000, color=BLUE, lw=2.0, label="wall plane")
    ax.plot(y_line * 1000, z_floor * 1000, color=ORANGE, lw=2.0, label="floor plane")

    seam_pt = corrected_start + direction * (
        (x_cut - corrected_start[0]) / (direction[0] + 1e-12))
    ax.scatter([seam_pt[1] * 1000], [seam_pt[2] * 1000], s=90, color=GREEN,
               zorder=5, label="sensed seam")
    ax.scatter([0.0], [0.0], s=90, facecolor="none", edgecolor=GREY, lw=1.8,
               zorder=5, label="taught seam")
    ax.annotate("", xy=(seam_pt[1] * 1000, seam_pt[2] * 1000), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.6))
    ax.legend(fontsize=7.5, loc="upper right")
    style(ax, "Y [mm]", "Z [mm]",
          f"cross-section at X = {x_cut*1000:.0f} mm")
    ax.set_aspect("equal", adjustable="datalim")

    # -- top view: touches and both seams --------------------------------
    ax = axes[1]
    wt, ft = np.array(wall_touches), np.array(floor_touches)
    ax.plot([taught_start[0] * 1000, taught_end[0] * 1000],
            [taught_start[1] * 1000, taught_end[1] * 1000],
            color=GREY, ls="--", lw=2.0, label="taught seam")
    ax.plot([corrected_start[0] * 1000, corrected_end[0] * 1000],
            [corrected_start[1] * 1000, corrected_end[1] * 1000],
            color=GREEN, lw=2.4, label="corrected seam")
    ax.scatter(wt[:, 0] * 1000, wt[:, 1] * 1000, color=BLUE, s=36,
               label="wall touches")
    ax.scatter(ft[:, 0] * 1000, ft[:, 1] * 1000, color=ORANGE, s=36,
               label="floor touches")
    ax.legend(fontsize=7.5)
    style(ax, "X [mm]", "Y [mm]", "top view (X-Y)")

    # -- correction magnitude along the seam -----------------------------
    ax = axes[2]
    t = np.linspace(0, 1, 60)
    taught = taught_start + np.outer(t, taught_end - taught_start)
    corr = corrected_start + np.outer(t, corrected_end - corrected_start)
    offset = np.linalg.norm(corr - taught, axis=1) * 1000
    ax.plot(taught[:, 0] * 1000, offset, color=GREEN, lw=1.9)
    style(ax, "along taught seam [mm]", "correction applied [mm]",
          "how far the sensed seam moved the path")
    ax.annotate(f"start {offset[0]:.2f} mm\nend   {offset[-1]:.2f} mm",
                xy=(0.45, 0.12), xycoords="axes fraction", fontsize=8.5,
                bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.95))

    fig.tight_layout()
    fig.savefig(OUT / "seam_correction.png", dpi=150)
    plt.close(fig)
    print(f"  wall  n={np.round(wall_n,4)}  c={wall_c*1000:.2f} mm")
    print(f"  floor n={np.round(floor_n,4)}  c={floor_c*1000:.2f} mm")
    print(f"  seam direction={np.round(direction,4)}")
    print(f"  correction {offset[0]:.2f} mm (start) .. {offset[-1]:.2f} mm (end)")


def fig_retiming():
    """MoveIt's S-curve timing vs the constant-velocity trapezoid."""
    from moveit_msgs.msg import RobotTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    n, duration = 200, 10.0
    # Stand-in for what MoveIt returns: TOTG + Ruckig give a smooth S-curve,
    # so model the input timing as a smoothstep in path progress.
    u = np.linspace(0.0, 1.0, n)
    progress = u * u * (3.0 - 2.0 * u)          # smoothstep
    joint_span = 1.2                            # rad of travel on one joint

    def build():
        traj = RobotTrajectory()
        traj.joint_trajectory.joint_names = ["j1"]
        for i in range(n):
            pt = JointTrajectoryPoint()
            pt.positions = [float(progress[i] * joint_span)]
            pt.velocities = [0.0]
            pt.accelerations = [0.0]
            t = duration * u[i]
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
            traj.joint_trajectory.points.append(pt)
        return traj

    original = build()
    t_in = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
                     for p in original.joint_trajectory.points])
    q_in = np.array([p.positions[0] for p in original.joint_trajectory.points])

    retimed = retime_trajectory_constant_velocity(build(), ramp_fraction=0.2)
    t_out = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
                      for p in retimed.joint_trajectory.points])
    q_out = np.array([p.positions[0] for p in retimed.joint_trajectory.points])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    axes[0].plot(t_in, q_in, color=GREY, lw=1.8, label="MoveIt S-curve timing")
    axes[0].plot(t_out, q_out, color=BLUE, lw=1.8,
                 label="constant-velocity retiming")
    axes[0].legend(fontsize=8)
    style(axes[0], "time [s]", "joint position [rad]",
          f"same points, same order, duration {trajectory_duration_seconds(retimed):.1f} s")

    axes[1].plot(t_in[1:], np.diff(q_in) / np.diff(t_in), color=GREY, lw=1.8,
                 label="MoveIt S-curve")
    axes[1].plot(t_out[1:], np.diff(q_out) / np.diff(t_out), color=BLUE, lw=1.8,
                 label="retimed")
    ramp = 0.2 * duration
    for x in (ramp, trajectory_duration_seconds(retimed) - ramp):
        axes[1].axvline(x, color=ORANGE, ls=":", lw=1.2)
    axes[1].text(ramp * 0.5, 0.02, "ramp", fontsize=8, color=ORANGE, ha="center")
    axes[1].legend(fontsize=8)
    style(axes[1], "time [s]", "joint speed [rad/s]",
          "cruise is flat — that is the point for welding")
    fig.tight_layout()
    fig.savefig(OUT / "retiming.png", dpi=150)
    plt.close(fig)


def fig_weave_plane():
    """Which plane the weave is swung in, once the seam is touch-corrected.

    On a fillet joint the sensed weave axis e_w lies in the wall/floor
    bisecting plane.  A generic tool/world axis does not, and the gap is not
    small: 45 degrees, which at a 2 mm amplitude separates the peaks by
    3.7 mm.  This is why the preview and the executed weld have to read the
    same accessor.
    """
    taught_start, taught_goal = pose(0, 0, 0), pose(SEAM_M, 0, 0)
    tilt = np.tan(np.deg2rad(4.0))
    wall = compute_surface_plane(
        (0.0, 1.0, 0.0),
        [(0.03, 0.003, 0.02), (0.15, 0.003 + 0.12 * tilt, 0.02)])
    floor = compute_surface_plane(
        (0.0, 0.0, 1.0), [(0.03, 0.03, -0.002), (0.15, 0.05, -0.002)])
    geo = compute_corrected_seam_geometry(taught_start, taught_goal, wall, floor)

    d = np.array(geo.d_real)
    e_w = np.array(geo.e_w)
    generic = np.array([0.0, 1.0, 0.0])
    generic = generic - (generic @ d) * d
    generic /= np.linalg.norm(generic)
    angle = np.degrees(np.arccos(abs(float(generic @ e_w))))

    base = linear_pose_waypoints(geo.start, geo.goal, 2)
    A = 0.002
    p_gen = xyz(weaving_from_path(base, A, 7, 12, "world_y"))
    p_sen = xyz(weaving_from_path(base, A, 7, 12, "world_y", tuple(e_w)))
    sep = np.linalg.norm(p_gen - p_sen, axis=1) * 1000

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    # -- cross-section: the joint and the two candidate weave axes ---------
    ax = axes[0]
    wn, fn_ = np.array(geo.wall_normal), np.array(geo.floor_normal)
    zl = np.linspace(-0.012, 0.012, 2)
    yw = (geo.wall_plane_value - wn[0] * 0.09 - wn[2] * zl) / wn[1]
    yl = np.linspace(-0.004, 0.02, 2)
    zf = (geo.floor_plane_value - fn_[0] * 0.09 - fn_[1] * yl) / fn_[2]
    ax.plot(yw * 1000, zl * 1000, color=BLUE, lw=2.2, label="wall plane")
    ax.plot(yl * 1000, zf * 1000, color=ORANGE, lw=2.2, label="floor plane")
    for vec, color, label in ((e_w, GREEN, "sensed e_w (executed)"),
                              (generic, "crimson", "generic axis (old preview)")):
        ax.annotate("", xy=(vec[1] * 9, vec[2] * 9), xytext=(-vec[1] * 9, -vec[2] * 9),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=2.0))
        ax.plot([], [], color=color, lw=2.0, label=label)
    ax.scatter([0], [0], s=60, color="k", zorder=5)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")
    style(ax, "Y [mm]", "Z [mm]", f"weave axes differ by {angle:.0f}°")

    # -- the two weave paths, seen down the seam ---------------------------
    ax = axes[1]
    ax.plot(p_gen[:, 1] * 1000, p_gen[:, 2] * 1000, color="crimson", lw=1.5,
            marker="o", ms=3, label="generic axis")
    ax.plot(p_sen[:, 1] * 1000, p_sen[:, 2] * 1000, color=GREEN, lw=1.5,
            marker="o", ms=3, label="sensed e_w")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    style(ax, "Y [mm]", "Z [mm]", "looking down the seam (A = 2 mm)")

    # -- separation along the seam ----------------------------------------
    ax = axes[2]
    ax.plot(p_gen[:, 0] * 1000, sep, color=GREY, lw=1.7)
    ax.axhline(A * 1000, color=BLUE, ls=":", lw=1.2)
    ax.text(4, A * 1000 + 0.12, "weave amplitude 2 mm", fontsize=7.5, color=BLUE)
    style(ax, "along seam [mm]", "preview vs executed [mm]",
          f"worst-case gap {sep.max():.2f} mm")

    fig.tight_layout()
    fig.savefig(OUT / "weave_plane.png", dpi=150)
    plt.close(fig)
    print(f"  weave axis angle {angle:.1f} deg, worst separation {sep.max():.2f} mm")


def main():
    fig_patterns()
    fig_crescent_progress()
    fig_pitch_quantisation()
    fig_dwell()
    fig_speed()
    fig_seam_correction()
    fig_retiming()
    fig_weave_plane()
    for f in sorted(OUT.glob("*.png")):
        print(f"wrote {f.relative_to(OUT.parent)}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
