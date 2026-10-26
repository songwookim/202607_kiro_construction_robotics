from glob import glob

from setuptools import setup

package_name = "hongik_scan_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy", "scipy", "pyyaml"],
    python_requires=">=3.10",
    zip_safe=False,
    maintainer="DBCM Lab (Hongik University)",
    maintainer_email="jm68112@gmail.com",
    description="Hongik 1G precision-sensing ROS 2 layer for KIRO: hongik_auto_teach_server (KIRO action "
    "/hongik_fine_sensing/precision_sensing, /hongik_fine_sensing/auto_teach and /hongik_fine_sensing/post_weld_scan), "
    "hongik_profiler_node, hongik_auto_teach_client (specimen input), hongik_post_weld_scan, MoveIt/JTC platform adapter (moveit_jtc).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "hongik_profiler_node = hongik_scan_ros.profiler_node:main",
            "hongik_auto_teach_server = hongik_scan_ros.auto_teach_server:main",
            "hongik_auto_teach_client = hongik_scan_ros.auto_teach_client:main",
            "hongik_post_weld_scan = hongik_scan_ros.post_weld_client:main",
            "hongik_precision_sensing = hongik_scan_ros.precision_sensing_client:main",
        ],
    },
)
