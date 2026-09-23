# KIRO Construction Robotics

ROS 2 Humble packages for the dual-arm KIRO construction robot.

The complete node/topic/service/action and RB controller data flow is documented
in [`docs/ROS_GRAPH.md`](docs/ROS_GRAPH.md). Path-generation equations, MoveIt
conversion, speed scaling, and ros2_control diagrams are in
[`docs/MOTION_MATH_AND_CONTROL_FLOW.md`](docs/MOTION_MATH_AND_CONTROL_FLOW.md).

[`docs/WELD_MATH_VISUAL.md`](docs/WELD_MATH_VISUAL.md) covers the maths that
document does not — crescent and circular weaving, dwell placement, weave speed
conversion, constant-velocity retiming, touch-based seam correction, and the
imitation-learning seam frame — as equations with plots. Every plot is produced
by calling the production functions, so regenerating them after a change is how
you check the document still matches the code: