#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/conversions.h>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(
      "straight_line_moveit_test",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  std::string group;
  std::string axis;
  double distance;
  bool execute;
  double speed;
  node->get_parameter_or("group", group, std::string("right_manipulator"));
  node->get_parameter_or("axis", axis, std::string("y"));
  node->get_parameter_or("distance", distance, 0.05);
  node->get_parameter_or("execute", execute, false);
  node->get_parameter_or("speed", speed, 0.05);

  if (group != "left_manipulator" && group != "right_manipulator")
  {
    RCLCPP_ERROR(node->get_logger(), "Unsupported group: %s", group.c_str());
    rclcpp::shutdown();
    return 2;
  }
  if (axis != "x" && axis != "y" && axis != "z")
  {
    RCLCPP_ERROR(node->get_logger(), "Axis must be x, y, or z");
    rclcpp::shutdown();
    return 2;
  }
  if (distance == 0.0 || speed <= 0.0 || speed > 1.0)
  {
    RCLCPP_ERROR(node->get_logger(), "Distance must be nonzero and speed in (0, 1]");
    rclcpp::shutdown();
    return 2;
  }

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  moveit::planning_interface::MoveGroupInterface move_group(node, group);
  move_group.setEndEffectorLink(
      group == "right_manipulator" ? "right_manipulator_ee_point"
                                   : "left_manipulator_ee_point");
  move_group.setMaxVelocityScalingFactor(speed);
  move_group.setMaxAccelerationScalingFactor(speed);

  const geometry_msgs::msg::Pose start = move_group.getCurrentPose().pose;
  geometry_msgs::msg::Pose target = start;
  if (axis == "x")
    target.position.x += distance;
  else if (axis == "y")
    target.position.y += distance;
  else
    target.position.z += distance;

  std::vector<geometry_msgs::msg::Pose> waypoints{ target };
  moveit_msgs::msg::RobotTrajectory trajectory;
  const double fraction =
      move_group.computeCartesianPath(waypoints, 0.005, 0.0, trajectory, true);

  RCLCPP_INFO(
      node->get_logger(),
      "Cartesian line: start=(%.4f, %.4f, %.4f), target=(%.4f, %.4f, %.4f), "
      "fraction=%.3f, points=%zu",
      start.position.x,
      start.position.y,
      start.position.z,
      target.position.x,
      target.position.y,
      target.position.z,
      fraction,
      trajectory.joint_trajectory.points.size());

  int exit_code = 1;
  if (fraction >= 0.999 && !trajectory.joint_trajectory.points.empty())
  {
    auto display_publisher =
        node->create_publisher<moveit_msgs::msg::DisplayTrajectory>(
            "/display_planned_path", rclcpp::QoS(1).transient_local());
    moveit_msgs::msg::DisplayTrajectory display;
    display.model_id = move_group.getRobotModel()->getName();
    if (const auto current_state = move_group.getCurrentState(2.0))
      moveit::core::robotStateToRobotStateMsg(
          *current_state, display.trajectory_start);
    display.trajectory.push_back(trajectory);
    display_publisher->publish(display);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    if (!execute)
    {
      RCLCPP_INFO(
          node->get_logger(),
          "Plan-only succeeded for %s; published /display_planned_path",
          group.c_str());
      exit_code = 0;
      executor.cancel();
      spinner.join();
      rclcpp::shutdown();
      return exit_code;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_ = trajectory;
    const auto result = move_group.execute(plan);
    if (result == moveit::core::MoveItErrorCode::SUCCESS)
    {
      const auto final_pose = move_group.getCurrentPose().pose;
      RCLCPP_INFO(
          node->get_logger(),
          "Execution succeeded: final=(%.4f, %.4f, %.4f)",
          final_pose.position.x,
          final_pose.position.y,
          final_pose.position.z);
      exit_code = 0;
    }
    else
    {
      RCLCPP_ERROR(node->get_logger(), "Execution failed: code=%d", result.val);
    }
  }
  else
  {
    RCLCPP_ERROR(node->get_logger(), "Incomplete Cartesian path; not executing");
  }

  executor.cancel();
  spinner.join();
  rclcpp::shutdown();
  return exit_code;
}
