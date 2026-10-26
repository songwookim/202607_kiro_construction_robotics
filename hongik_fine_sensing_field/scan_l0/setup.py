from setuptools import setup

package_name = "scan_l0"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    package_data={package_name: ["jobs/*.yaml", "data/*.jsonl"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy>=1.21,<2", "scipy>=1.8", "pyyaml>=5.4"],   # + scan_core, vc3dz_driver (this tree)
    python_requires=">=3.10",
    zip_safe=False,
    maintainer="DBCM Lab",
    maintainer_email="jm68112@gmail.com",
    description="Pure-Python groove sensing run (1G core) for the VC nano 3D-Z profiler (the robot platform is the MoveIt adapter in "
    "hongik_scan_ros); depends on scan_core and vc3dz_driver.",
    license="Apache-2.0",
    tests_require=["pytest"],
)
