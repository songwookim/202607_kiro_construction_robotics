"""Hongik 1G precision sensing on a MoveIt / ros2_control platform (KIRO) - the sensing layer only.

Already running on the platform (KIRO): move_group (planning group, planning frame), the JointTrajectoryController, robot_state_publisher
(/joint_states -> TF) with the robot model that includes our EOAT (sensor + torch geometry), and /clock with Isaac Sim. This launch adds,
all in the namespace /hongik_fine_sensing:
  * static TF  laser_frame -> hongik_profiler_optical_frame (xyz / rpy of the EOAT calibration file; config laser_frame / tool_frame)
  * hongik_auto_teach_server   action servers precision_sensing (KIRO welding_interfaces/action/PrecisionSensing) and auto_teach
                               (hongik_scan_msgs/action/AutoTeach), progress lines on ~/log; it owns the sensor during a run
  * (option with_profiler_node) hongik_profiler_node - streams the sensor only on request (~/start); the server takes the sensor over
                               (~/stop) before a goal
Nothing here commands or configures the robot controller: KIRO's main system sends the first command (the goal) and the run moves
the arm through KIRO's move_group and JTC only.

  ros2 launch hongik_scan_ros hongik_moveit.launch.py platform_config:=/path/hongik_kiro.yaml calib_file:=/path/eoat_kiro_01.yaml \
      job_file:=/path/job_current.yaml
  ros2 launch hongik_scan_ros hongik_moveit.launch.py ... job_overrides:='{weld_plan.stick_out_mm: 17.5, precision.end_position_mm: 0.2}'
      # site-level run-time job overrides (server parameter job_overrides, a string; every goal of both actions; scan_l0.job_overrides)
use_sim_time follows use_sim_time of the platform YAML (the same value for all our nodes).

NumPy / SciPy: our nodes run with the system python3 and the apt pair this package is built and tested with (MANIFEST: python3-numpy
1.21.5, python3-scipy 1.8.0; apt SciPy is compiled for NumPy 1.x). A host with another NumPy earlier on python3's path (KIRO
2026-10-14: NumPy 2 in ~/.local and /usr/local) would break SciPy. So our nodes get PYTHONNOUSERSITE=1 and numeric_site (default
the apt directory) APPENDED to PYTHONPATH: PYTHONPATH entries come before /usr/local in sys.path, the ROS and workspace entries keep
their order, and every child process (the G1 worker, pools) inherits it. Nothing else on the host changes. numeric_site:='' leaves
the environment as it is.
"""
import os

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

NS = "hongik_fine_sensing"


def _params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    return dict(d["/**"]["ros__parameters"]) if "/**" in d else dict(d)


def _static_tf_args(calib_path: str, parent: str, child: str) -> list:
    with open(calib_path, "r", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    if c.get("parent_frame") != "laser_link":
        raise RuntimeError(f"calib file {calib_path} must store laser_link -> optical (got {c.get('parent_frame')!r})")
    x, y, z = (float(v) for v in c["xyz"])
    r, p, yw = (float(v) for v in c["rpy"])
    return ["--x", repr(x), "--y", repr(y), "--z", repr(z), "--roll", repr(r), "--pitch", repr(p), "--yaw", repr(yw),
            "--frame-id", parent, "--child-frame-id", child]


def numeric_env(numeric_site: str) -> dict:
    """Environment of our nodes (module docstring 'NumPy / SciPy'): {} when numeric_site is '' (unchanged), else PYTHONNOUSERSITE=1 and
    numeric_site appended to the current PYTHONPATH (once)."""
    site = str(numeric_site or "").strip()
    if not site:
        return {}
    parts = [q for q in os.environ.get("PYTHONPATH", "").split(os.pathsep) if q]
    if site not in parts:
        parts.append(site)
    return {"PYTHONNOUSERSITE": "1", "PYTHONPATH": os.pathsep.join(parts)}


def launch_setup(context, *args, **kwargs):
    cfg_path = LaunchConfiguration("platform_config").perform(context)
    calib = LaunchConfiguration("calib_file").perform(context)
    p = _params(cfg_path)
    if str(p.get("motion_backend")) != "moveit_jtc" or str(p.get("pose_source")) != "ros_tf":
        raise RuntimeError(f"{cfg_path}: hongik_moveit.launch.py needs pose_source ros_tf + motion_backend moveit_jtc")
    use_sim_time = bool(p.get("use_sim_time"))
    laser, optical = str(p["laser_frame"]), str(p["tool_frame"])
    common = {"use_sim_time": use_sim_time}
    server_params = {
        **common, "platform_config": cfg_path, "calib_file": calib,
        "confirm_mode": LaunchConfiguration("confirm_mode").perform(context),
        "kiro_confirm_mode": LaunchConfiguration("kiro_confirm_mode").perform(context),
        "out_root": LaunchConfiguration("out_root").perform(context),
        "sensor_ip": LaunchConfiguration("sensor_ip").perform(context),
        "sensor_port": int(LaunchConfiguration("sensor_port").perform(context) or 0),
        "sensor_init_overrides": LaunchConfiguration("sensor_init_overrides").perform(context),
        "local_speed_mps": float(LaunchConfiguration("local_speed_mps").perform(context)),
        "transit_speed_mps": float(LaunchConfiguration("transit_speed_mps").perform(context)),
        "descend_speed_mps": float(LaunchConfiguration("descend_speed_mps").perform(context)),
        "dwell_frames": int(LaunchConfiguration("dwell_frames").perform(context)),
        "probe_frames": int(LaunchConfiguration("probe_frames").perform(context)),
        # site-level run-time job overrides (scan_l0.job_overrides): applied to every goal of both actions - a STRING parameter (a YAML /
        # JSON mapping as text: launch_ros would otherwise parse it as YAML and refuse a mapping)
        "job_overrides": ParameterValue(LaunchConfiguration("job_overrides").perform(context), value_type=str),
    }
    job_file = LaunchConfiguration("job_file").perform(context)
    if job_file:
        server_params["job_file"] = job_file
    env = numeric_env(LaunchConfiguration("numeric_site").perform(context))
    nodes = [
        Node(package="tf2_ros", executable="static_transform_publisher", name="hongik_profiler_optical_tf", namespace=NS,
             output="log", arguments=_static_tf_args(calib, laser, optical), parameters=[common]),
        # The server names its own node. A global __node remap would also rename
        # its internal MoveIt adapter and make action introspection count it twice.
        Node(package="hongik_scan_ros", executable="hongik_auto_teach_server", namespace=NS,
             output="screen", parameters=[server_params], additional_env=env),
    ]
    if LaunchConfiguration("with_profiler_node").perform(context) == "true":
        nodes.append(Node(package="hongik_scan_ros", executable="hongik_profiler_node", name="hongik_profiler_node", namespace=NS,
                          output="screen", parameters=[{**common, "platform_config": cfg_path, "frame_id": optical,
                                                        "sensor_ip": server_params["sensor_ip"],
                                                        "sensor_port": server_params["sensor_port"]}], additional_env=env))
    return nodes


def generate_launch_description():
    share = get_package_share_directory("hongik_scan_ros")
    return LaunchDescription([
        DeclareLaunchArgument("platform_config", default_value=os.path.join(share, "config", "hongik_kiro.yaml")),
        DeclareLaunchArgument("calib_file", default_value=os.path.join(get_package_share_directory("eoat_description"), "calib",
                                                                       "eoat_kiro_01.yaml"),
                              description="EOAT calibration YAML (laser_link -> optical, x_sign, dt_s, g1 block); the default is the "
                                          "Hongik lab calibration with KIRO's mounting yaw (eoat_kiro_01.yaml) - replace it with the "
                                          "file measured on KIRO's robot"),
        DeclareLaunchArgument("confirm_mode", default_value="each", description="each | large | scan | auto | none (~/confirm_move)"),
        DeclareLaunchArgument("kiro_confirm_mode", default_value="none",
                              description="confirmation of KIRO PrecisionSensing goals: none (default) | auto | scan | large | each"),
        DeclareLaunchArgument("out_root", default_value=os.path.expanduser("~/scan_data/hongik_auto_teach")),
        DeclareLaunchArgument("job_file", default_value="", description="job YAML (specimen, tolerances, process; written by "
                              "hongik_auto_teach_client --setup); '' = the template groove_10T_1G"),
        DeclareLaunchArgument("sensor_ip", default_value="", description="'' = platform config sensor.ip"),
        DeclareLaunchArgument("sensor_port", default_value="0", description="0 = platform config sensor.port"),
        DeclareLaunchArgument("sensor_init_overrides", default_value="", description="INIT_STANDARD overrides 'Name=value,...'"),
        DeclareLaunchArgument("local_speed_mps", default_value="0.0", description="0 = the platform TCP limit (moveit_jtc.limits.max_tcp_speed_mps)"),
        DeclareLaunchArgument("transit_speed_mps", default_value="0.0", description="0 = the platform TCP limit (moveit_jtc.limits.max_tcp_speed_mps)"),
        DeclareLaunchArgument("descend_speed_mps", default_value="0.0", description="0 = the platform TCP limit (moveit_jtc.limits.max_tcp_speed_mps)"),
        DeclareLaunchArgument("dwell_frames", default_value="0", description="0 = config auto_teach.dwell_frames"),
        DeclareLaunchArgument("probe_frames", default_value="30"),
        DeclareLaunchArgument("with_profiler_node", default_value="false"),
        DeclareLaunchArgument("numeric_site", default_value="/usr/lib/python3/dist-packages",
                              description="directory of the apt numpy / scipy our nodes must use (MANIFEST), appended to PYTHONPATH with "
                                          "PYTHONNOUSERSITE=1 so a NumPy 2 in ~/.local or /usr/local cannot shadow it; '' = leave the "
                                          "environment unchanged"),
        DeclareLaunchArgument("job_overrides", default_value="",
                              description="site-level run-time job overrides for every goal (both actions; KIRO's site adjusts this way): a "
                                          "YAML / JSON mapping of dotted job paths, e.g. job_overrides:='{precision.end_position_mm: 0.2, "
                                          "weld_plan.stick_out_mm: 17.5}'; checked at start; '' = none (scan_l0.job_overrides)"),
        OpaqueFunction(function=launch_setup),
    ])
