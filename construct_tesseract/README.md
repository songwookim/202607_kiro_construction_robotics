# KIRO Tesseract validation

This package keeps the normal KIRO workspace buildable when Tesseract is not
installed. When a Tesseract overlay is sourced, it also builds
`environment_check`, which loads the real KIRO URDF and SRDF.

The versions in `tesseract_humble.repos` match the upstream ROS 2 Humble
0.22.x release line.

## Build the Tesseract overlay

```bash
mkdir -p /home/irs/ros2_ws/src/tesseract_ws/src
cd /home/irs/ros2_ws/src/tesseract_ws
vcs import --shallow src < /home/irs/ros2_ws/src/construct_robot_ros2/construct_tesseract/tesseract_humble.repos
/home/irs/ros2_ws/src/construct_robot_ros2/construct_tesseract/scripts/patch_arm64.sh \
  /home/irs/ros2_ws/src/tesseract_ws/src
rosdep install --from-paths src --ignore-src -iry --rosdistro humble
source /opt/ros/humble/setup.bash
colcon build --base-paths src \
  --packages-skip tesseract_qt tesseract_rviz \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  --parallel-workers 2
```

This builds the complete headless planning stack, including TrajOpt, Task
Composer, monitoring, the planning server and ROS examples. The optional Qt
and RViz plugins additionally require the Graphviz development libraries:

```bash
sudo apt-get install libgraphviz-dev
colcon build --base-paths src \
  --packages-select tesseract_qt tesseract_rviz \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
```

## Validate the KIRO model

```bash
source /opt/ros/humble/setup.bash
source /home/irs/ros2_ws/install/setup.bash
source /home/irs/ros2_ws/src/tesseract_ws/install/setup.bash
cd /home/irs/ros2_ws
colcon build --packages-select construct_tesseract
source install/setup.bash
source /home/irs/ros2_ws/src/tesseract_ws/install/setup.bash
export TESSERACT_RESOURCE_PATH=/home/irs/ros2_ws/install
ros2 run construct_tesseract environment_check \
  /home/irs/ros2_ws/install/construct_description/share/construct_description/urdf_0528/construct_robot_0528.urdf \
  /home/irs/ros2_ws/install/construct_tesseract/share/construct_tesseract/config/construct_robot_0528_tesseract.srdf
```

The Tesseract-specific SRDF keeps `dual_arm` as an explicit joint list because
Tesseract 0.22 does not expand MoveIt subgroup-only compound groups. MoveIt uses
its normal SRDF with left/right subgroups so RViz exposes both end-effector
interactive markers.

After IPOPT is installed, build through `tesseract_motion_planners` and use
the upstream `FreespacePipeline` or OMPL examples as the starting point for
the first trajectory-planning integration.

## Run the KIRO planning check

The check plans a collision-aware five-state left-arm motion with OMPL and
then optimizes the seed with TrajOpt:

```bash
source /opt/ros/humble/setup.bash
source /home/irs/ros2_ws/install/setup.bash
source /home/irs/ros2_ws/src/tesseract_ws/install/setup.bash
export TESSERACT_RESOURCE_PATH=/home/irs/ros2_ws/install
ros2 run construct_tesseract motion_planning_check \
  /home/irs/ros2_ws/install/construct_description/share/construct_description/urdf_0528/construct_robot_0528.urdf \
  /home/irs/ros2_ws/install/construct_tesseract/share/construct_tesseract/config/construct_robot_0528_tesseract.srdf
```

## Visualize the dual-arm Tesseract trajectory

Keep the MoveIt RViz launch running, then publish the coordinated 14-joint
trajectory:

```bash
source /opt/ros/humble/setup.bash
source /home/irs/ros2_ws/install/setup.bash
source /home/irs/ros2_ws/src/tesseract_ws/install/setup.bash
export TESSERACT_RESOURCE_PATH=/home/irs/ros2_ws/install
ros2 run construct_tesseract dual_arm_rviz_demo \
  /home/irs/ros2_ws/install/construct_description/share/construct_description/urdf_0528/construct_robot_0528.urdf \
  /home/irs/ros2_ws/install/construct_tesseract/share/construct_tesseract/config/construct_robot_0528_tesseract.srdf
```

For the constrained cooperative example, both TCPs act as if they grasp one
rigid member. Tesseract KDL IK solves nine Cartesian states while preserving
the initial left-TCP-to-right-TCP transform:

```bash
source /opt/ros/humble/setup.bash
source /home/irs/ros2_ws/install/setup.bash
source /home/irs/ros2_ws/src/tesseract_ws/install/setup.bash
export TESSERACT_RESOURCE_PATH=/home/irs/ros2_ws/install
ros2 run construct_tesseract dual_arm_constrained_demo \
  /home/irs/ros2_ws/install/construct_description/share/construct_description/urdf_0528/construct_robot_0528.urdf \
  /home/irs/ros2_ws/install/construct_tesseract/share/construct_tesseract/config/construct_robot_0528_tesseract.srdf
```

RViz receives the trajectory on `/display_planned_path` and cyan constraint
rungs on `/tesseract_constraint_markers`.
