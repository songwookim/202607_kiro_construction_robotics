# VS Code breakpoint debugging

Open `/home/irs/ros2_ws` as the VS Code workspace. The workspace `.vscode`
directory contains the launch tasks and attach configurations.

## Start a debug session

1. Set breakpoints in the source tree.
2. Run the `ROS2: launch weld debug (SAFE)` task.
3. In Run and Debug, start `Attach: weld GUI + Cartesian server`.
4. Stop both attach sessions before stopping the ROS launch task.

The SAFE task disables motion execution and selects fake hardware. Use
`ROS2: launch weld debug (REAL)` only with the physical robot area attended.

The debug endpoints are:

- `weld_action_gui` and `hicomm_welder.py`: `127.0.0.1:5678`
- `cartesian_path_server`: `127.0.0.1:5679`

Build, launch, and both debug children use
`/home/irs/ros2_ws/.venv/bin/python` (Python 3.10). The workspace source script
removes Conda Python 3.14 from `PATH`, because ROS 2 Humble native modules and
generated interfaces require the CPython 3.10 ABI.

For a terminal session outside VS Code, run:

```bash
cd /home/irs/ros2_ws
source src/construct_robot_ros2/scripts/use_ros_python.bash
```

Pausing inside `hicomm_welder.py` also pauses its cyclic network thread. Do
not leave a breakpoint paused there while live welding equipment expects its
40 ms Hi-COMM cycle.

Python modules are installed with `--symlink-install`, so ordinary Python
edits do not need a rebuild. Rebuild after changing launch files, interfaces,
package metadata, URDF, or installed configuration files.
