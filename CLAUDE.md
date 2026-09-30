Read CLAUDE.md.

First refactoring step only:

Extract `WeldGuiNode` from
`construct_robot/construct_robot/gui/weld_action_gui.py`
to
`construct_robot/construct_robot/nodes/weld_runtime_node.py`.

Preserve behavior exactly.
Do not redesign GUI↔ROS communication, threading, safety logic,
ROS interfaces, or execution semantics.

Inspect only the target implementation, its direct dependencies,
and relevant tests. Avoid an exhaustive repository scan.

Update imports, run relevant tests/build checks, and report the result.
Do not start the next refactoring step.