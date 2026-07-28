#include <iostream>
#include <memory>
#include <string>

#include <Eigen/Core>
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
#include <tesseract_motion_planners/trajopt/profile/trajopt_default_composite_profile.h>
#include <tesseract_motion_planners/trajopt/profile/trajopt_default_plan_profile.h>
#include <tesseract_motion_planners/trajopt/trajopt_motion_planner.h>

using namespace tesseract_planning;

int main(int argc, char** argv)
{
  if (argc < 3 || argc > 5)
  {
    std::cerr << "usage: motion_planning_check <robot.urdf> <robot.srdf> "
                 "[left_manipulator|right_manipulator] "
                 "[ompl|ompl-trajopt]\n";
    return 2;
  }

  auto locator = std::make_shared<tesseract_common::GeneralResourceLocator>();
  auto environment = std::make_shared<tesseract_environment::Environment>();
  if (!environment->init(
          tesseract_common::fs::path(argv[1]),
          tesseract_common::fs::path(argv[2]),
          locator))
  {
    std::cerr << "environment initialization failed\n";
    return 1;
  }

  const std::string group = argc == 4 ? argv[3] : "right_manipulator";
  if (group != "left_manipulator" && group != "right_manipulator")
  {
    std::cerr << "unsupported planning group: " << group << "\n";
    return 2;
  }
  const std::string planner_mode = argc == 5 ? argv[4] : "ompl";
  if (planner_mode != "ompl" && planner_mode != "ompl-trajopt")
  {
    std::cerr << "unsupported planner mode: " << planner_mode << "\n";
    return 2;
  }
  const auto joint_group = environment->getJointGroup(group);
  const auto& joint_names = joint_group->getJointNames();
  Eigen::VectorXd start(6);
  if (group == "right_manipulator")
    start << -3.14, 0.314159, 1.43117, 1.1002556, 0.261799, 2.89725;
  else
    start << 0.0, -0.7767975217522227, -1.570825302024956,
        -0.00004466603319160641, -0.7767317140952323,
        0.00006233535385690629;
  Eigen::VectorXd goal = start;
  const double direction = 1.0;
  goal[0] += direction * 0.05;

  StateWaypointPoly start_waypoint{ StateWaypoint(joint_names, start) };
  MoveInstruction start_instruction(
      start_waypoint, MoveInstructionType::FREESPACE, "DEFAULT");

  CompositeInstruction program;
  tesseract_common::ManipulatorInfo manipulator;
  manipulator.manipulator = group;
  manipulator.working_frame = "World";
  manipulator.tcp_frame = group == "right_manipulator"
                              ? "right_manipulator_ee_point"
                              : "left_manipulator_ee_point";
  program.setManipulatorInfo(manipulator);
  program.appendMoveInstruction(start_instruction);
  for (int index = 1; index <= 4; ++index)
  {
    Eigen::VectorXd point = start;
    point[0] += direction * 0.0125 * static_cast<double>(index);
    StateWaypointPoly waypoint{ StateWaypoint(joint_names, point) };
    program.appendMoveInstruction(
        MoveInstruction(waypoint, MoveInstructionType::FREESPACE, "DEFAULT"));
  }

  const auto environment_state = environment->getState();
  CompositeInstruction interpolated =
      generateInterpolatedProgram(program, environment_state, environment, 3.14, 1.0, 3.14);

  constexpr const char* ompl_namespace = "OMPLMotionPlannerTask";
  constexpr const char* trajopt_namespace = "TrajOptMotionPlannerTask";
  auto profiles = std::make_shared<ProfileDictionary>();
  profiles->addProfile<OMPLPlanProfile>(
      ompl_namespace, "DEFAULT", std::make_shared<OMPLDefaultPlanProfile>());
  profiles->addProfile<TrajOptPlanProfile>(
      trajopt_namespace, "DEFAULT", std::make_shared<TrajOptDefaultPlanProfile>());
  profiles->addProfile<TrajOptCompositeProfile>(
      trajopt_namespace,
      "DEFAULT",
      std::make_shared<TrajOptDefaultCompositeProfile>());

  PlannerRequest request;
  request.instructions = interpolated;
  request.env = environment;
  request.env_state = environment_state;
  request.profiles = profiles;

  OMPLMotionPlanner ompl(ompl_namespace);
  PlannerResponse ompl_response = ompl.solve(request);
  if (!ompl_response)
  {
    std::cerr << "OMPL failed: " << ompl_response.message << "\n";
    return 1;
  }

  if (planner_mode == "ompl")
  {
    const auto trajectory = toJointTrajectory(ompl_response.results);
    std::cout << "Tesseract OMPL planning succeeded\n"
              << "group=" << group << "\n"
              << "ompl_steps=" << trajectory.size() << "\n"
              << "start=" << start.transpose() << "\n"
              << "goal=" << goal.transpose() << "\n";
    return 0;
  }

  request.instructions = ompl_response.results;
  TrajOptMotionPlanner trajopt(trajopt_namespace);
  PlannerResponse trajopt_response = trajopt.solve(request);
  if (!trajopt_response)
  {
    std::cerr << "TrajOpt failed: " << trajopt_response.message << "\n";
    return 1;
  }

  const auto trajectory = toJointTrajectory(trajopt_response.results);
  std::cout << "Tesseract planning succeeded\n"
            << "group=" << group << "\n"
            << "ompl_steps=" << ompl_response.results.size() << "\n"
            << "trajopt_steps=" << trajectory.size() << "\n"
            << "start=" << start.transpose() << "\n"
            << "goal=" << goal.transpose() << "\n";
  return 0;
}
