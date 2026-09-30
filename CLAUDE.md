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

## Package layout

The PySide6 GUI (`gui_qt/`) and the top-level compatibility shims were removed.
`construct_robot/construct_robot/` holds only `__init__.py` and the subpackages
above; import from the owning subpackage (`test_package_boundaries.py` enforces this).

## Validation

For production changes:
- run relevant tests
- verify production console entry points
- run `colcon build --packages-select construct_robot`

Unit tests do not replace real-hardware validation.