# KIRO Construction Robotics

ROS 2 Humble packages for the dual-arm KIRO construction robot.

The complete node/topic/service/action and RB controller data flow is documented
in [`docs/ROS_GRAPH.md`](docs/ROS_GRAPH.md). Path-generation equations, MoveIt
conversion, speed scaling, and ros2_control diagrams are in
Every plot is produced
by calling the production functions, so regenerating them after a change is how
you check the document still matches the code: