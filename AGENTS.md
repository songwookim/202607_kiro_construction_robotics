# AGENTS.md

## Project

ROS 2 Humble dual-arm robotic welding system using:
- MoveIt 2
- ros2_control
- Rainbow Robotics / RBPodo
- Hi-COMM welder
- Fastech touch sensing
- Tkinter welding GUI

Main application:
- `construct_robot/construct_robot/weld_action_gui.py`

Motion/math helpers:
- `construct_robot/construct_robot/cartesian_path_common.py`
- `construct_robot/construct_robot/cartesian_path_server.py`

Configuration:
- `construct_moveit_config/`

Tests:
- `construct_robot/test/`

Useful docs:
- `docs/ROS_GRAPH.md` for ROS/control data flow
- `docs/MOTION_MATH_AND_CONTROL_FLOW.md` for motion/control math
- `docs/WELD_MATH_VISUAL.md` for seam/weave geometry

Do not read all docs for every task. Read only what is relevant.

## Environment

This workspace uses ROS 2 Humble / Python 3.10.

Never edit generated files under:
- `build/`
- `install/`
- `log/`

Edit source files only.

## Change Policy

- Inspect only the files relevant to the requested change.
- Make the smallest coherent change.
- Do not perform unrelated refactors.
- Reuse existing functions/state instead of creating parallel implementations.
- Preserve unrelated behavior and existing tests.
- Do not silently change robot motion, welding, controller, or safety behavior.

## Robot Safety

- Planning, preview, teaching, seam correction, and loading a pass must never start welding.
- Never enable the arc unless the requested execution workflow explicitly requires it.
- Preserve collision, touch-guard, and clearance checks.
- Do not introduce automatic robot motion without an existing explicit workflow.
- Be careful around controller switching and command discontinuities.
- Preview and execution must use the same path/geometric assumptions.

## Robot State / Control

Measured state:

RB control box
-> RBPodo hardware
-> ros2_control state interface
-> joint_state_broadcaster
-> `/joint_states`

Commands:

MoveIt
-> JointTrajectoryController
-> ros2_control position command interface
-> RBPodoHardwareInterface
-> `move_servo_j`
-> RB control box

Never assume `/joint_states.position` array order.
Resolve joints using `/joint_states.name`.

## Seam / Weave Geometry

When touch-corrected geometry exists, use the sensed seam-local geometry rather
than arbitrary World/Tool axes.

Important frame quantities:
- `d_real`: corrected seam direction
- `e_w`: weave transverse direction
- `e_a`: approach direction

Keep preview and execution consistent.

Do not replace existing seam geometry with World-axis assumptions unless the
task explicitly requests it.

## Multi-pass Welding

Current production workflow is sequential pass-wise correction.

The selected pass is the current correction anchor:
- correcting Pass 1 updates 1,2,3,4
- correcting Pass 2 keeps 1 and updates 2,3,4
- correcting Pass 3 keeps 1,2 and updates 3,4
- correcting Pass 4 updates only 4

Corrections are cumulative.
Later corrections must operate on the CURRENT corrected state, not restart from
the original source coordinates.

Keep these concepts separate:
1. immutable source/reference data
2. current corrected/predicted multi-pass state
3. newly measured START/GOAL registration

Use `correct_remaining_passes()` as the core propagation logic where applicable.

Source welding logs must never be overwritten.

Saved/current pass teaching and corrected pass YAML files may be updated
separately from immutable source logs.

Preserve source provenance/hash validation where applicable.

A successful correction must update the active multi-pass state used by:
- corrected START/GOAL verification
- selected-pass loading
- subsequent correction
- path generation
- welding execution

## Multi-pass Teaching

START and GOAL registration are separate measurements.

START/GOAL registration may change:
- translation
- seam direction
- propagated later-pass positions/orientations

WAIT poses remain safe transition poses and must not be confused with welding
START/GOAL poses.

Reuse the existing keyboard teaching path; do not add competing keyboard
listeners.

## GUI

`weld_action_gui.py` currently owns GUI orchestration and much of the welding
workflow.

Avoid duplicating geometry or multi-pass state merely to update the GUI.

GUI controls must reflect the actual working state, not maintain an independent
copy of correction data.

Long robot/network operations must not block the Tkinter UI thread.

## Tests

Run focused tests first.

Multi-pass:
`pytest construct_robot/test/test_four_pass_correction.py -q`

Weave/seam:
`pytest construct_robot/test/test_weld_weave.py -q`

Motion/math:
`pytest construct_robot/test/test_cartesian_path_math.py -q`

Run broader regression tests only when appropriate.

For behavior changes:
- update/add focused tests
- verify unrelated relevant tests still pass
- report exactly which tests were run

Do not claim hardware validation from unit tests.

## Working Style

For each task:
1. inspect the relevant current implementation
2. modify it directly
3. run the smallest relevant test set
4. summarize changed files and behavior

Do not produce a long design document unless requested.
Do not scan the whole repository unless the task actually requires it.