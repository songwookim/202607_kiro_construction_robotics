# Project Instructions

## Architecture

Production GUI: Tkinter.

Package roles:

- `gui/`: Tkinter UI and view glue
- `application/`: workflow/orchestration
- `core/`: pure calculation, domain state, validation
- `nodes/`: ROS 2 runtime
- `io/`: hardware/file adapters

Preferred dependency direction:

GUI → Application → Core
                  ↘ Nodes / IO

## Refactoring policy

Prefer:
move → extract → preserve

Do not rewrite working behavior without a clear reason.

Preserve:
- ROS topic/service/action names
- launch behavior
- ARC synchronization
- STOP behavior
- touch-guard behavior
- keyboard deadman behavior
- multipass semantics
- threading/event synchronization unless explicitly requested

## Qt policy

`construct_robot/construct_robot/gui_qt/` is frozen.

Do not:
- modify Qt/PySide6 code
- synchronize Qt with Tk
- run Qt-specific tests
- remove Qt-only compatibility shims

unless explicitly requested.

## Validation

For production changes:
- run relevant non-Qt tests
- verify production console entry points
- run `colcon build --packages-select construct_robot`

Unit tests do not replace real-hardware validation.