from setuptools import find_packages, setup

package_name = "scan_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["numpy>=1.21,<2", "scipy>=1.8", "pyyaml>=5.4"],   # tested: tools/python_env.yaml
    extras_require={"h5": ["h5py"], "test": ["pytest<8"]},
    python_requires=">=3.10",
    zip_safe=True,
    maintainer="DBCM Lab",
    maintainer_email="jm68112@gmail.com",
    description="ROS-independent scan algorithms for the RB20 + VC nano 3D-Z line-profiler scan stack "
    "(path generation, safety envelope, counter clock, reconstruction, quality, calibration, synthetic data).",
    license="MIT",
    tests_require=["pytest"],
)
