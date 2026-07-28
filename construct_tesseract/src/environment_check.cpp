#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include <tesseract_common/resource_locator.h>
#include <tesseract_environment/environment.h>
#include <tesseract_kinematics/core/joint_group.h>

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "usage: environment_check <robot.urdf> <robot.srdf>\n";
    return 2;
  }

  auto locator = std::make_shared<tesseract_common::GeneralResourceLocator>();
  tesseract_environment::Environment environment;
  if (!environment.init(
          tesseract_common::fs::path(argv[1]),
          tesseract_common::fs::path(argv[2]),
          locator))
  {
    std::cerr << "Tesseract failed to initialize the KIRO environment\n";
    return 1;
  }

  std::cout << "Tesseract environment initialized\n"
            << "root_link=" << environment.getRootLinkName() << '\n'
            << "links=" << environment.getLinkNames().size() << '\n'
            << "joints=" << environment.getJointNames().size() << '\n';

  for (const std::string& group : { "left_manipulator", "right_manipulator", "dual_arm" })
  {
    try
    {
      auto joint_group = environment.getJointGroup(group);
      std::cout << "group=" << group << " joints=" << joint_group->numJoints() << '\n';
    }
    catch (const std::exception& exception)
    {
      std::cerr << "group=" << group << " unavailable: " << exception.what() << '\n';
      return 1;
    }
  }

  return 0;
}
