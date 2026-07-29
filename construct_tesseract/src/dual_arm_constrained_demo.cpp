#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tesseract_common/resource_locator.h>
#include <tesseract_environment/environment.h>
#include <tesseract_kinematics/core/kinematic_group.h>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace
{
Eigen::VectorXd closestSolution(const tesseract_kinematics::IKSolutions& solutions,
                                const Eigen::VectorXd& seed)
{
  if (solutions.empty())
    throw std::runtime_error("IK returned no solution");
  auto best = solutions.front();
  double distance = (best - seed).squaredNorm();
  for (const auto& solution : solutions)
  {
    const double candidate = (solution - seed).squaredNorm();
    if (candidate < distance)
    {
      best = solution;
      distance = candidate;
    }
  }
  return best;
}

Eigen::VectorXd valuesFor(const std::vector<std::string>& names,
                          const std::unordered_map<std::string, double>& values)
{
  Eigen::VectorXd result(static_cast<Eigen::Index>(names.size()));
  for (Eigen::Index i = 0; i < result.size(); ++i)
    result[i] = values.at(names[static_cast<std::size_t>(i)]);
  return result;
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  if (argc != 3)
  {
    std::cerr << "usage: dual_arm_constrained_demo <robot.urdf> <robot.srdf>\n";
    return 2;
  }
  auto node = std::make_shared<rclcpp::Node>("tesseract_dual_arm_constrained_demo");
  auto locator = std::make_shared<tesseract_common::GeneralResourceLocator>();
  auto env = std::make_shared<tesseract_environment::Environment>();
  if (!env->init(tesseract_common::fs::path(argv[1]),
                 tesseract_common::fs::path(argv[2]), locator))
  {
    RCLCPP_ERROR(node->get_logger(), "Environment initialization failed");
    return 1;
  }

  const auto left = env->getKinematicGroup("left_manipulator");
  const auto right = env->getKinematicGroup("right_manipulator");
  const auto dual = env->getJointGroup("dual_arm");
  const auto& dual_names = dual->getJointNames();
  const auto left_names = left->getJointNames();
  const auto right_names = right->getJointNames();

  std::unordered_map<std::string, double> home{
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
  env->setState(home);
  const auto initial_state = env->getState();
  const Eigen::Isometry3d left_world =
      initial_state.link_transforms.at("left_manipulator_ee_point");
  const Eigen::Isometry3d right_world =
      initial_state.link_transforms.at("right_manipulator_ee_point");
  // Hard cooperative constraint: the transform from left TCP to right TCP
  // stays constant, as if both tools grasp one rigid construction member.
  const Eigen::Isometry3d left_to_right = left_world.inverse() * right_world;

  Eigen::VectorXd left_seed = valuesFor(left_names, home);
  Eigen::VectorXd right_seed = valuesFor(right_names, home);
  std::vector<Eigen::VectorXd> path;
  tesseract_common::AlignedVector<Eigen::Isometry3d> left_tcp_path;
  tesseract_common::AlignedVector<Eigen::Isometry3d> right_tcp_path;
  double max_position_error = 0.0;
  double max_rotation_error = 0.0;

  for (int step = 0; step <= 8; ++step)
  {
    const double phase = static_cast<double>(step) / 8.0;
    Eigen::Isometry3d target_left = left_world;
    target_left.pretranslate(Eigen::Vector3d(
        0.035 * std::sin(M_PI * phase), 0.0, 0.08 * std::sin(M_PI * phase)));
    const Eigen::Isometry3d target_right = target_left * left_to_right;

    left_seed = closestSolution(
        left->calcInvKin(
            tesseract_kinematics::KinGroupIKInput(
                target_left, "World", "left_manipulator_ee_point"),
            left_seed),
        left_seed);
    right_seed = closestSolution(
        right->calcInvKin(
            tesseract_kinematics::KinGroupIKInput(
                target_right, "World", "right_manipulator_ee_point"),
            right_seed),
        right_seed);

    std::unordered_map<std::string, double> state_values = home;
    for (std::size_t i = 0; i < left_names.size(); ++i)
      state_values[left_names[i]] = left_seed[static_cast<Eigen::Index>(i)];
    for (std::size_t i = 0; i < right_names.size(); ++i)
      state_values[right_names[i]] = right_seed[static_cast<Eigen::Index>(i)];
    env->setState(state_values);
    const auto state = env->getState();
    const auto actual_left = state.link_transforms.at("left_manipulator_ee_point");
    const auto actual_right = state.link_transforms.at("right_manipulator_ee_point");
    left_tcp_path.push_back(actual_left);
    right_tcp_path.push_back(actual_right);
    const auto relative_error = left_to_right.inverse() * actual_left.inverse() * actual_right;
    max_position_error = std::max(max_position_error, relative_error.translation().norm());
    max_rotation_error = std::max(
        max_rotation_error,
        Eigen::AngleAxisd(relative_error.rotation()).angle());
    path.push_back(valuesFor(dual_names, state_values));
  }

  if (max_position_error > 0.002 || max_rotation_error > 0.01)
  {
    RCLCPP_ERROR(node->get_logger(),
                 "Constraint violation: position=%g m rotation=%g rad",
                 max_position_error, max_rotation_error);
    return 1;
  }

  moveit_msgs::msg::DisplayTrajectory display;
  display.model_id = "construct_robot_0528";
  display.trajectory_start.joint_state.name = dual_names;
  display.trajectory_start.joint_state.position.assign(
      path.front().data(), path.front().data() + path.front().size());
  moveit_msgs::msg::RobotTrajectory trajectory;
  trajectory.joint_trajectory.joint_names = dual_names;
  for (std::size_t i = 0; i < path.size(); ++i)
  {
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions.assign(path[i].data(), path[i].data() + path[i].size());
    const double seconds = 0.7 * static_cast<double>(i);
    point.time_from_start.sec = static_cast<std::int32_t>(seconds);
    point.time_from_start.nanosec =
        static_cast<std::uint32_t>((seconds - std::floor(seconds)) * 1e9);
    trajectory.joint_trajectory.points.push_back(std::move(point));
  }
  display.trajectory.push_back(std::move(trajectory));

  visualization_msgs::msg::MarkerArray markers;
  visualization_msgs::msg::Marker constraint;
  constraint.header.frame_id = "World";
  constraint.ns = "tesseract_rigid_constraint";
  constraint.id = 0;
  constraint.type = visualization_msgs::msg::Marker::LINE_LIST;
  constraint.action = visualization_msgs::msg::Marker::ADD;
  constraint.scale.x = 0.018;
  constraint.color.r = 0.1F;
  constraint.color.g = 0.9F;
  constraint.color.b = 1.0F;
  constraint.color.a = 1.0F;
  for (std::size_t i = 0; i < left_tcp_path.size(); ++i)
  {
    geometry_msgs::msg::Point lp;
    lp.x = left_tcp_path[i].translation().x();
    lp.y = left_tcp_path[i].translation().y();
    lp.z = left_tcp_path[i].translation().z();
    geometry_msgs::msg::Point rp;
    rp.x = right_tcp_path[i].translation().x();
    rp.y = right_tcp_path[i].translation().y();
    rp.z = right_tcp_path[i].translation().z();
    constraint.points.push_back(lp);
    constraint.points.push_back(rp);
  }
  markers.markers.push_back(constraint);

  auto trajectory_pub = node->create_publisher<moveit_msgs::msg::DisplayTrajectory>(
      "/display_planned_path", rclcpp::QoS(1).transient_local().reliable());
  auto marker_pub = node->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/tesseract_constraint_markers", rclcpp::QoS(1).transient_local().reliable());
  for (int repeat = 0; repeat < 3; ++repeat)
  {
    trajectory_pub->publish(display);
    marker_pub->publish(markers);
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  RCLCPP_INFO(node->get_logger(),
              "Constrained dual-arm plan: points=%zu, max relative error=%.6f m / %.6f rad",
              path.size(), max_position_error, max_rotation_error);
  rclcpp::shutdown();
  return 0;
}
