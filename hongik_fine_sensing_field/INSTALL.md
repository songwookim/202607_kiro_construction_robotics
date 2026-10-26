# Hongik 1G precision sensing - installation

Target: Ubuntu 22.04, ROS 2 Humble, the system Python 3.10 (apt numpy 1.21 / scipy 1.8 / PyYAML 5.4). No virtual environment and no
pip packages are needed. The package tree (this directory) and KIRO's `welding_interfaces` are built with a standard `colcon build`.

## 1. Prerequisites

| What | Where it comes from |
|---|---|
| ROS 2 Humble (`ros-humble-ros-base` or desktop) with `moveit_msgs`, `control_msgs`, `tf2_ros`, `launch_ros`, `xacro` | apt (`rosdep install`, step 3) |
| KIRO's robot stack running on the same PC: `move_group`, the JointTrajectoryController, `robot_state_publisher` with a robot model that includes our EOAT (README §2) | KIRO |
| Nothing robot-controller specific: no teach pendant, control box, controller mode or controller settings are used by this package | - |
| KIRO's `welding_interfaces` (PrecisionSensing action) | KIRO - in the same workspace or in an underlay workspace |
| A network interface in `192.168.3.x` for the VC nano 3D-Z sensor (factory IP `192.168.3.15`, TCP 1096) | PC setup (NetworkManager) |
| CPU: the 1G compute worker uses min(cores - 4, 16) child processes at the lowest priority | - |

## 2. Workspace

```bash
mkdir -p ~/hongik_ws/src && cd ~/hongik_ws/src
cp -r <this directory> hongik_fine_sensing          # or a git checkout / symlink
cp -r <KIRO's welding_interfaces> welding_interfaces  # skip if an underlay workspace already provides it
```

KIRO's `welding_interfaces/package.xml` declares no `<export><build_type>`. colcon then treats it as a catkin package and the workspace
setup does not register it (`ros2 interface show` fails). Write a `colcon.meta` override once (the package's own files are not
changed; the tool applies the same rule to any `ament_cmake` package without a declared build type):

```bash
cd ~/hongik_ws
python3 src/hongik_fine_sensing/tools/colcon_meta_ament_cmake.py      # writes ./colcon.meta
```

(If KIRO adds `<export><build_type>ament_cmake</build_type></export>` to their package.xml, the tool finds nothing to do.)

## 3. Dependencies and build

```bash
source /opt/ros/humble/setup.bash
source <KIRO workspace>/install/setup.bash          # if move_group / welding_interfaces come from KIRO's workspace
cd ~/hongik_ws
rosdep install --from-paths src --ignore-src -y     # python3-numpy, python3-scipy, python3-yaml, ROS message packages
colcon build
source install/setup.bash
```

Packages: `vc3dz_driver` (sensor driver), `scan_core` (estimation, outputs), `scan_l0` (1G run), `hongik_scan_msgs`, `hongik_scan_ros`
(server, launch, config), `eoat_description` (EOAT xacro, mesh, calibration file).

## 4. Checks (no robot motion)

```bash
ros2 interface show welding_interfaces/action/PrecisionSensing
ros2 run hongik_scan_ros hongik_auto_teach_server --help
ros2 run hongik_scan_ros hongik_precision_sensing --help
ros2 launch hongik_scan_ros hongik_moveit.launch.py --show-args
ping -c 3 192.168.3.15                                              # sensor LAN
ros2 run vc3dz_driver vc3dz --ip 192.168.3.15 backup                 # keep the sensor's original parameters (once)
ros2 run vc3dz_driver vc3dz --ip 192.168.3.15 read-params | head     # sensor answers (sends no laser start)
```

## 5. Tests (optional, no robot, no sensor)

```bash
sudo apt install python3-pytest     # if missing
cd ~/hongik_ws/src/hongik_fine_sensing
for t in vc3dz_driver/tests scan_core/tests scan_l0/tests eoat_description/test \
         "hongik_scan_ros/test/test_kiro_io.py hongik_scan_ros/test/test_package_scope.py hongik_scan_ros/test/test_hongik_units.py"; do
    python3 -m pytest -q $t || break      # one pytest per package: each test directory has its own conftest.py
done
```

`scan_l0/tests/test_sim_teach1g.py` runs complete 1G sensing runs on a virtual arm with a simulated sensor (several minutes).

Next: README.md (configuration, calibration, running, interface, outputs).

## 6. Python virtual environment (optional: tools, tests, `python3 -m ...`)

The ROS nodes do not use a venv: their installed entry scripts start with `#!/usr/bin/python3`, so the server always runs with the
system Python and the apt packages of section 3 (numpy 1.21.5, scipy 1.8.0, PyYAML 5.4.1 - the versions in the delivery's
MANIFEST.txt). Build the workspace with any venv deactivated. numpy 2 is not supported.

From the YAML spec (`tools/python_env.yaml`; profiles `system` = the apt versions, nothing installed, works offline; `pinned` = the
same versions installed inside the venv; `lab` = the Hongik lab development versions):

```bash
python3 tools/make_venv.py --profile system --dest ~/venv_hongik
source ~/venv_hongik/bin/activate
```

The tool checks the result (Python 3.10, numpy < 2, scipy, PyYAML, the three packages importable) and writes
`~/venv_hongik/hongik_venv_report.txt`. Profiles that install packages need pip in the venv (apt `python3-venv python3-pip`, or
`--pip-from <site-packages with pip>`, or `--get-pip` to download get-pip.py).

From the packages' `setup.py` (needs pip; the dependencies are declared with the tested bounds `numpy>=1.21,<2`, `scipy>=1.8`,
`pyyaml>=5.4` and are already satisfied by the system packages in a `--system-site-packages` venv):

```bash
python3 -m venv --system-site-packages ~/venv_hongik
~/venv_hongik/bin/pip install ./scan_core ./vc3dz_driver ./scan_l0
```

Check which numpy a Python sees: `python3 -c "import numpy; print(numpy.__version__, numpy.__file__)"`. A copy in `~/.local` or a
`PYTHONPATH` entry in front of the system one shadows the apt package for the ROS nodes too - remove it or set `PYTHONNOUSERSITE=1`.
