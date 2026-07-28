#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>

#include <Eigen/Core>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tesseract_command_language/composite_instruction.h>
#include <tesseract_command_language/move_instruction.h>
#include <tesseract_command_language/profile_dictionary.h>
#include <tesseract_command_language/state_waypoint.h>
#include <tesseract_command_language/utils.h>
#include <tesseract_common/resource_locator.h>
#include <tesseract_environment/environment.h>
#include <tesseract_kinematics/core/joint_group.h>
#include <tesseract_motion_planners/core/types.h>
#include <tesseract_motion_planners/core/utils.h>
#include <tesseract_motion_planners/ompl/ompl_motion_planner.h>
#include <tesseract_motion_planners/ompl/profile/ompl_default_plan_profile.h>
#include <tesseract_motion_planners/simple/interpolation.h>

using namespace tesseract_planning;

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  if (argc != 3)
  {
    std::cerr << "usage: dual_arm_rviz_demo <robot.urdf> <robot.srdf>\n";
    return 2;
  }

  auto node = std::make_shared<rclcpp::Node>("tesseract_dual_arm_rviz_demo");
  auto locator = std::make_shared<tesseract_common::GeneralResourceLocator>();
  auto environment = std::make_shared<tesseract_environment::Environment>();
  if (!environment->init(
          tesseract_common::fs::path(argv[1]),
          tesseract_common::fs::path(argv[2]),
          locator))
  {
    RCLCPP_ERROR(node->get_logger(), "Tesseract environment initialization failed");
    return 1;
  }

  const auto joint_group = environment->getJointGroup("dual_arm");
  const auto& names = joint_group->getJointNames();
  std::map<std::string, double> home{
    { "left_manipulator_joint1", 0.0 },
    { "left_manipulator_joint2", -0.7767975217522227 },
    { "left_manipulator_joint3", -1.570825302024956 },
    { "left_manipulator_joint4", 0.0 },
    { "left_manipulator_joint5", -0.7767317140952323 },
    { "left_manipulator_joint6", 0.0 },
    { "right_manipulator_joint1", 3.0 },
    { "right_manipulator_joint2", 0.314159 },
    { "right_manipulator_joint3", 1.43117 },
    { "right_manipulator_joint4", 1.1002556 },
    { "right_manipulator_joint5", 0.261799 },
    { "right_manipulator_joint6", 2.89725 },
    { "robot_head_rev_joint1", 0.0 },
    { "robot_head_rev_joint2", 0.0 },
  };
  std::map<std::string, double> delta{
    { "left_manipulator_joint1", 0.08 },
    { "left_manipulator_joint2", 0.06 },
    { "left_manipulator_joint4", -0.10 },
    { "left_manipulator_joint5", 0.04 },
    { "right_manipulator_joint1", -0.08 },
    { "right_manipulator_joint2", -0.06 },
    { "right_manipulator_joint4", 0.10 },
    { "right_manipulator_joint5", -0.04 },
    { "robot_head_rev_joint1", 0.05 },
  };

  Eigen::VectorXd start(static_cast<Eigen::Index>(names.size()));
  Eigen::VectorXd goal(static_cast<Eigen::Index>(names.size()));
  for (Eigen::Index index = 0; index < start.size(); ++index)
  {
    const auto& name = names[static_cast<std::size_t>(index)];
    start[index] = home.at(name);
    goal[index] = start[index] + delta[name];
  }

  CompositeInstruction program;
  tesseract_common::ManipulatorInfo manipulator;
  manipulator.manipulator = "dual_arm";
  manipulator.working_frame = "World";
  manipulator.tcp_frame = "left_manipulator_ee_point";
  program.setManipulatorInfo(manipulator);
  for (int step = 0; step <= 6; ++step)
  {
    const double ratio = static_cast<double>(step) / 6.0;
    Eigen::VectorXd point = start + ratio * (goal - start);
    StateWaypointPoly waypoint{ StateWaypoint(names, point) };
    program.appendMoveInstruction(
        MoveInstruction(waypoint, MoveInstructionType::FREESPACE, "DEFAULT"));
  }

  const auto state = environment->getState();
  CompositeInstruction seed =
      generateInterpolatedProgram(program, state, environment, 3.14, 1.0, 3.14);
  // The Tesseract Simple planner/interpolator preserves all coordinated
  // dual-arm waypoints. OMPL tends to simplify these small coupled segments
  // into independent two-state subproblems, which is less useful for an RViz
  // choreography demonstration.
  const auto planned = toJointTrajectory(seed);
  moveit_msgs::msg::DisplayTrajectory display;
  display.model_id = "construct_robot_0528";
  display.trajectory_start.joint_state.name = names;
  display.trajectory_start.joint_state.position.assign(start.data(), start.data() + start.size());
  moveit_msgs::msg::RobotTrajectory robot_trajectory;
  robot_trajectory.joint_trajectory.joint_names = names;
  for (std::size_t index = 0; index < planned.size(); ++index)
  {
    trajectory_msgs::msg::JointTrajectoryPoint point;
    const auto& state_point = planned.states[index];
    point.positions.assign(
        state_point.position.data(),
        state_point.position.data() + state_point.position.size());
    const double seconds = 0.8 * static_cast<double>(index);
    point.time_from_start.sec = static_cast<std::int32_t>(std::floor(seconds));
    point.time_from_start.nanosec = static_cast<std::uint32_t>(
        (seconds - std::floor(seconds)) * 1e9);
    robot_trajectory.joint_trajectory.points.push_back(std::move(point));
  }
  display.trajectory.push_back(std::move(robot_trajectory));

  auto publisher = node->create_publisher<moveit_msgs::msg::DisplayTrajectory>(
      "/display_planned_path", rclcpp::QoS(1).transient_local().reliable());
  for (int repeat = 0; repeat < 3; ++repeat)
  {
    publisher->publish(display);
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  RCLCPP_INFO(
      node->get_logger(),
      "Published Tesseract dual-arm trajectory: joints=%zu, points=%zu",
      names.size(),
      planned.size());
  rclcpp::shutdown();
  return 0;
}
