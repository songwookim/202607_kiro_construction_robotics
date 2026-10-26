# Hongik 1G precision sensing (KIRO package)

Laser-profiler sensing of a 1G groove before welding: our VC nano 3D-Z (12/24) large sensor on KIRO's robot measures the seam and
returns the cross sections and a 4-pass weld plan.

KIRO's main system sends the first command - a `PrecisionSensing` goal (`welding_interfaces`) to our ROS 2 action server - and our
server runs: it moves the robot only through KIRO's MoveIt (`move_group`) and JointTrajectoryController, then returns the result.
The package has no robot-controller concepts at all: no controller or program mode, no arm activation, no controller status, no tool /
payload / collision-sensitivity / speed settings, no teach pendant. The robot, its controller and its own safety functions are KIRO's;
when a goal arrives the robot is expected to be ready to move through KIRO's MoveIt / JTC.

Build and install: [INSTALL.md](INSTALL.md).

## 1. What runs

`ros2 launch hongik_scan_ros hongik_moveit.launch.py ...` starts (namespace `/hongik_fine_sensing`):

| Node | Role |
|---|---|
| `hongik_profiler_optical_tf` | static TF `laser_link -> hongik_profiler_optical_frame` from the calibration file (`xyz`, `rpy`) |
| `hongik_auto_teach_server` | action servers `precision_sensing` (KIRO), `auto_teach` (our console client) and `post_weld_scan` (on-demand post-weld raw-data scan, section 6); owns the sensor during a run; progress lines on `~/log` |
| `hongik_profiler_node` (option `with_profiler_node:=true`) | streams sensor profiles on request (`~/start`, `~/stop`) for inspection |

Run sequence of one 1G goal (about 3-5 min): the arm pose at goal acceptance is the start pose -> MoveIt joint-space move to look at
the first approximate point -> straight descent -> static dwell + correction -> streaming transit along the seam -> dwell at the other
point -> streaming passes (alternating direction, laser pitch -5 deg) until every delivered quantity has k sigma <= its tolerance or
the pass budget (`auto_teach.max_passes`) is used -> outputs.

Robot pose: every `/joint_states` message is turned into the flange pose by forward kinematics with the URDF on
`/robot_description` (joints matched by name) - TF is not read per sample (it may arrive late); static transforms (flange -> optical,
flange -> TCP, the gravity frame) are read from TF once and checked against the URDF, and once per goal the FK pose is compared with
TF `base <- flange` at the same stamp (a different robot model refuses the goal). Whether `/joint_states` carries measured encoder
angles or the controller's command echo is told apart from the recorded data and reported in the run log.

Safety of this package = planning, before anything is sent: our motion envelope (every straight leg: distance of every tool point - torch tip, sensor
housing, cable - to the plate plane and the support below it, clamp height, laser tilt, transit height; margin = `safety.clearance_m`
+ stopping distance) and MoveIt (collision-checked Cartesian paths and joint-space plans, including the two objects this package adds
to the planning scene: `hongik/specimen` and `hongik/floor`, derived from the approximate points + the specimen design, then from the
measured specimen). The weld plan is checked on this arm before it is delivered (D14, no motion): per pass the torch orientation
closest to the WPS travel / work angles that is collision-free (the measured specimen at the weld pose: plates, groove, backing and
the plan's convex beads of the earlier passes; the gas nozzle and contact tip may come to 0 mm of it, every other tool part keeps
k x the combined sigma) and reachable along the whole pass (MoveIt Cartesian check); a plan that fails is not delivered (code 4, with the reason). A plan starts from a `/joint_states` message published after the request, and a trajectory is sent only after a
state published after planning has arrived (none within `tf_timeout_s` = nothing sent).

There is no runtime motion monitor and no sensor-based clearance stop: during a motion only the JTC result is read - a trajectory
the JTC rejects or aborts ends the run as a motion failure (`ABORTED`, code 1) without any further command (when no JTC result
arrives at all within 1.5 x the trajectory time + `goal_time_margin_s`, our own goal is cancelled). The only stop of this
package is the action cancel (decelerating stop of our trajectory, result `ABORTED`). The robot's own protective functions are
KIRO's. A `/joint_states` gap during a scan leg is not a stop: the leg's frames are reconstructed from the poses received (frames
outside them drop out and the leg's data checks decide); if `/joint_states` is still missing when the leg ends, its arrival cannot be
confirmed and the run ends as a motion failure.

## 2. What KIRO provides

| Item | Details |
|---|---|
| Robot model in KIRO's MoveIt (URDF + SRDF) **including our EOAT** | `construct_robot_0528_hongik_v2.urdf` r4 (Hongik 2026-10-13: KIRO's construct_robot_0528 with our 1st bracket + torch + VC nano 3D-Z on `right_manipulator_link6` as identified from KIRO's September weld logs - mounted at eoat_yaw -90 deg, the torch turned 180 deg in its clamp so its neck bends away from the sensor: `hongik_flange` -> `hongik_eoat_base_link` (collision mesh of bracket, torch and sensor) -> `hongik_laser_link`, `hongik_weld_link`; the TCP `right_manipulator_ee_point` = our torch's contact-tip end, nominal CAD: flange -> TCP (-0.45, 237.86, 297.17) mm, rpy (0, -30, -90) deg). The TCP sits 238 mm off the flange axis: KIRO's on-site wire-tip calibration of `right_ee_point_joint` matters (0.1 deg about the flange axis = 0.4 mm at the tip). In the SRDF disable the collision pair `hongik_eoat_base_link` - `right_manipulator_link6` (the adapter plate sits on the flange). The optical frame is not in the URDF: our launch publishes it (section 1). |
| Frame / group / controller names | `base_frame`, `flange_frame`, MoveIt group, JTC action, controller state topic -> `hongik_kiro.yaml` [T2] [T4] [T5] |
| Robot-specific joint configuration | `auto_teach.wrist2_branch_1g` [T8]: our lab RB20 restricts 1G to wrist-2 negative (the configuration its calibration was made in); set it for KIRO's arm after the calibration there (`any` = no restriction) |
| Speed / acceleration limits and the stop deceleration of a cancel | `moveit_jtc.limits`, `stop_decel_*` [T7] |
| Ground frame in TF (gravity) | `gravity.frame` / `axis` [T3] |
| The hand-eye calibration of our sensor on KIRO's robot | section 5 (done together with Hongik on site) |
| The robot ready to move when a goal is sent | KIRO's controller, its modes and its safety; our server only plans with `move_group` and sends trajectories to the JTC |

This package adds only `hongik/specimen` and `hongik/floor` to KIRO's planning scene (`/apply_planning_scene`) and removes only those.
During the D14 check (no motion) it also adds the weld-pose specimen `hongik/weld/*` and lets the EOAT link (the link that carries the
gas nozzle and contact tip) touch those objects in the allowed-collision matrix (its parts are checked per role by our exact model);
both are removed / restored when the check ends.

## 3. Configuration

`hongik_scan_ros/config/hongik_kiro.yaml` (copy it, edit, pass `platform_config:=`). Fields to fill on KIRO's PC:

| Mark | Field | Note |
|---|---|---|
| [T1] | `use_sim_time` | true only with Isaac Sim `/clock` |
| [T2] | `base_frame`, `flange_frame`, `eoat_frame`, `laser_frame`, `torch_frame` | names in KIRO's TF with the v2 robot model (`World`, `hongik_flange`, `hongik_eoat_base_link`, `hongik_laser_link`, `right_manipulator_ee_point`; check with `ros2 run tf2_ros tf2_echo World hongik_laser_link`); the base is MoveIt's planning frame |
| [T3] | `gravity.frame`, `gravity.axis` | a ground frame in TF; the base-frame gravity is derived through TF |
| [T4] | `planning_group`, `moveit_jtc.move_group_name`, move_group services | the group `right_manipulator`; its tip link (e.g. `right_manipulator_ee_point`) and `flange_frame` must be rigidly connected (both fixed to `right_manipulator_link6` in the v2 model) |
| [T5] | `jtc_action`, `controller_name`, `controller_state_topic`, `joint_states_topic`, `robot_description_topic` | KIRO's ros2_control names |
| [T6] | `safety.tool_body_boxes` (+ `tool_body_mesh`, `torch_protrusion_m` as fallbacks), `weld_torch` | the envelope's tool geometry is read from the robot model at run time: the collision geometry mounted on `flange_frame` in `robot_description` + `tool_body_boxes` (parts the model does not hold), and the torch tip = the wire end = TF `flange_frame` <- `weld_torch.tcp_frame` + (the gas nozzle end, derived from the URDF collision geometry, + the job's wire stick-out `weld_plan.stick_out_mm`, measured from the nozzle end) along the torch axis. `weld_torch` = our torch (TCP `right_manipulator_ee_point` = the contact-tip end, recessed 1.2 mm inside the gas nozzle in the CAD); after KIRO's on-site TCP calibration only the URDF changes (D14 and the tool geometry read TF), `flange_from_tcp` is the record |
| [T7] | `moveit_jtc.limits`, `stop_decel_*` | KIRO's robot / controller specification |
| [T8] | `auto_teach.wrist2_branch_1g` | see section 2 |
| [T9] | `sensor.ip`, `sensor.port` | sensor LAN |

Specimen (design and tolerances) - the job file the server reads (`job_file:=`), written on this PC:

```bash
ros2 run hongik_scan_ros hongik_auto_teach_client --setup --job ~/scan_data/kiro/job_current.yaml
```

It asks "Reuse the previous inputs? [u]se / [d]o not use", then the root gap, bevel angle and weld length (the other values -
thickness, root face, backing, plate widths, tolerances, precision targets, the measurement speed, the weld process - come from the
template `scan_l0/jobs/groove_10T_1G.yaml`; `--edit-all` asks the specimen ones too). The template's specimen is KIRO's site specimen:
10T single bevel 30 deg, root gap 5 mm, backing 10T, weld length 200 mm, plates 150 mm each. The file is re-read for every goal. The run
measures at the job's `job.speed_mps` (template: 30 mm/s at 150 fps; `scan_defaults.speed_mps` holds the same value) and stops passing
when every delivered quantity reaches its 1-sigma precision target (`precision:`: ends 0.10 mm, root and section dimensions 0.05 mm) or
the pass budget is used. The weld process of the plan (`weld_plan:`) holds KIRO's September 2026 values: nominal torch angles travel
-42 / -41 / -44 / -44 deg (drag) and work +9 / -20 / +7 / +7 deg for root / fill 1 / fill 2 / cap (+ = the wire / torch axis toward
the work tilts toward the bevel face; the bevel was on the LEFT of KIRO's travel, so these are KIRO's logged work angles with the
sign turned), and the wire stick-out `stick_out_mm` 20 measured from the gas nozzle end - set it to KIRO's value.

The weld plan (scan_core.weld_plan, the plan file `weld_plan.json` schema 3):
- Beads are convex (scan_core.bead_geometry): each pass puts its mass-balance area (deposition efficiency x wire area x WFS / travel
  speed at the WPS settings) on the current surface - the measured section with the earlier passes' beads - as a level pool from
  its aim point bounded on top by a circular arc over its toe chord. One shape value: the bead toe angle
  `weld_plan.bead_toe_angle_deg` (default 30 deg, a literature value for as-welded GMAW beads; to be calibrated with post-weld
  scans of each pass): arc height / chord = tan(toe / 2) / 2 (0.134 at 30 deg); the woven cap spreads the same area over its weave
  width (flatter). A job file's older key `bead_aspect_ratio` (kappa) is read as the toe angle 2 atan(2 kappa); both keys = refused.
  The layer heights, the aim surface, D14's weld scene (each earlier pass's bead as a convex solid) and every output come from
  these beads.
- Aim (KIRO's practice, from the measured section; mirrored with the bevel side the scan resolves): the wire end meets the current
  surface; laterally root = the corner of the root face and the bevel face, fill 1 = the root bead's toe on the square wall, fill 2 =
  midway between fill 1's toes, cap = midway between the groove top edges. Layers outside this 4-pass practice (a further fill,
  multi-bead layers, several cap passes, other groove kinds) use the geometric rule (the bead's band centre); each pass records its
  rule and why (`weld_plan.json` passes `aim_lateral_rule` / `aim_why`; `kiro/pass_<k>_info.yaml` `aim_offset_mm.rule`). The old
  per-pass offset keys `aim_lateral_tcp_mm` / `aim_lateral_axis` are refused (a job file: with the reason; an override: unknown key).
- The cap reinforcement `weld_design.cap_reinforcement_mm` defaults to 1.87 +- 0.19 mm (KIRO's root + 2 fills + cap on the measured
  10T groove with these beads; the tolerance = deposition efficiency 0.90..1.00) - it chooses the number of layers.

## 4. Running

```bash
ros2 launch hongik_scan_ros hongik_moveit.launch.py \
    platform_config:=/path/hongik_kiro.yaml calib_file:=/path/eoat_kiro_02.yaml \
    job_file:=$HOME/scan_data/kiro/job_current.yaml
```

(`eoat_kiro_02.yaml` = the calibration measured on KIRO's robot, section 5; without `calib_file` the launch uses the shipped starting
point `eoat_kiro_01.yaml`.)

Other arguments: `out_root` (results, default `~/scan_data/hongik_auto_teach`), `kiro_confirm_mode` (default `none`; `each|large|scan|auto`
wait for `~/confirm_move` before moves), `sensor_ip` / `sensor_port` (override the config), speeds (`local_speed_mps`,
`transit_speed_mps`, `descend_speed_mps`; 0 = the platform TCP limit), `with_profiler_node`, `job_overrides`.

Site adjustments without editing the job file - `job_overrides` (server parameter, applied to every goal; the PrecisionSensing goal
has no field for it): a YAML / JSON mapping of dotted paths into the job document, e.g.

```bash
ros2 launch hongik_scan_ros hongik_moveit.launch.py ... \
    job_overrides:='{weld_plan.stick_out_mm: 17.5, precision.end_position_mm: 0.2, "weld_plan.travel_angle_deg": [-40, -40, -40]}'
```

Blocks `job`, `specimen`, `qc_targets`, `precision`, `weld_plan`, `weld_design`, `auto_teach`; a key must be one the job file, the
template or the block's parser knows. Keys a run does not take from the job (the goal points, frames, safety-policy values, the
robot-model torch, measured values) and unknown keys are refused by name: a wrong `job_overrides` stops the server at start (checked
against `job_file`), and a goal whose job file it does not fit is refused (code: input error) before anything moves. Every run records
the overrides (`verification/run_log.json` `job_overrides`: path, old, new, source), the effective job (`verification/job_effective.yaml`)
and the npz meta inputs. A measurement speed chosen this way (`job.speed_mps`) must lie inside the calibration's noise-prior speed range
(`g1.noise_priors.speed_range_mps` of the calibration file: 15..60 mm/s for the shipped lab priors), otherwise the goal is refused
before any motion. The TCP roll follows the WPS travel angle (`scan_core.weld_orient` NECK_RULE): the straight torch body leans the
same way as the nozzle along the travel and further than it (drag: both trail; push: both lead); a pass without a feasible orientation
in that roll is refused with the reason (no flip to the other roll).

Sending a goal from a terminal (operator; KIRO's main system sends the same goal programmatically) - in a second terminal with the
same environment:

```bash
ros2 run hongik_scan_ros hongik_precision_sensing --a X,Y,Z --b X,Y,Z    # the two approximate points [m] in World (the base frame)
ros2 run hongik_scan_ros hongik_precision_sensing --points FILE          # YAML {a: [x, y, z], b: [x, y, z], frame: World}
ros2 run hongik_scan_ros hongik_precision_sensing                        # "Reuse the previous points (A [..], B [..])? [u]se / [d]o not use"
```

| Argument | |
|---|---|
| `--a X,Y,Z`, `--b X,Y,Z` | the two points (`'X Y Z'` in quotes also); orientations are sent as identity |
| `--points FILE` | the two points from a file; `frame` optional, otherwise it must be the server's base frame (`World`) |
| (none) | the previous goal's points (`~/.hongik/precision_sensing_points.yaml`, saved for every accepted goal): use / do not use |
| `--server-timeout S`, `--log-topic NAME`, `--action NAME` | defaults 30 s, found automatically, `/hongik_fine_sensing/precision_sensing` |

It prints every progress line, Ctrl-C cancels the goal, and at the end prints the result code, message, number of paths, run folder,
`kiro/pass_<k>.yaml` files and `data_file_path`, and saves the full result as `<run folder>/precision_sensing_result.yaml`. Exit code
0 = SUCCESS, 1 = an error code or not sent, 2 = not started (points). There is no dry run (the goal has no field for one) and no
per-goal job adjustment: the client's goals run with the server's `job_overrides` like KIRO's.

Conditions for a goal (this release):
- `pose_a` / `pose_b`: approximate seam points, **positions** in the base frame (orientations are not used). Both must lie on the
  seam **inside** the specimen span - not beyond an end, not on the backing or a fixture. The run stops with `DATA_FAIL: groove not in view
  at the S rough point` (code 3) when the first dwell does not see the groove.
- Before the goal is sent the arm must stand in a posture clear of the specimen: the start pose is the arm pose at goal acceptance.
  If a robot link is inside `hongik/specimen` / `hongik/floor` at that moment the server refuses before any motion and names the link.
- One goal at a time; a second goal while one runs is aborted ("busy").

## 5. Calibration (redo on KIRO's robot)

File format (`eoat_description/calib/eoat_kiro_01.yaml` = the Hongik lab calibration of this sensor and bracket (`eoat_lab_01.yaml`)
with KIRO's mounting yaw -90 deg, the lab's - the launch default, a starting point only):

| Key | Meaning |
|---|---|
| `parent_frame: laser_link`, `xyz`, `rpy` | `laser_link -> optical frame` (the hand-eye result; published by our launch as a static TF) |
| `eoat_yaw` | flange -> `eoat_base_link` mount yaw of that robot (record; on KIRO's robot it is `hongik_eoat_base_joint` of the v2 URDF, and the server reads the flange -> optical / eoat transforms from TF) |
| `x_sign` | sign of the sensor x axis (+1 / -1), consistent with the `rpy` yaw |
| `dt_s`, `g1.dt_s`, `g1.dt_sigma_s` | sensor time offset to the robot pose stream and its sigma [s] (1G uses the `g1` block) |
| `g1.he_sigma_m` | sigma of the posture-dependent hand-eye residual [m] (plans the overscan) |

The hand-eye transform depends on how the EOAT sits on KIRO's flange, and the time offset depends on KIRO's `/joint_states` timing.
Both must be measured again with the sensor on KIRO's robot (Hongik does this on site with its development tools; plan about half a
day):

1. Check the mounting in TF: `tf2_echo hongik_flange hongik_laser_link` (the v2 model r4: yaw -90 deg). Start from `eoat_kiro_01.yaml`.
2. Sensor x sign: move the arm +20 mm along `eoat_base_link` +y over a plate edge; the edge moves -x in the profile when `x_sign` is -1.
3. Hand-eye: a box corner on the table measured statically from 10-20 postures (both torch sides, look-down 15-40 deg, two ranges);
   one rigid correction of the optical frame is fitted and cross-validated on held-out postures (lab result: scatter 4.8 -> 0.8 mm).
   Write the corrected `xyz` / `rpy`; `g1.he_sigma_m` = the held-out residual.
4. Time offset: on the specimen, the passes alternate direction; the ends read in the two directions differ by 2 v dt. Repeat runs
   and set `g1.dt_s` / `g1.dt_sigma_s` to their mean / spread (lab: 3.7 +- 1.0 ms with its `/joint_states`).
5. Verify: repeat the 1G run 3-5 times; the delivered ends and root line repeat within the job tolerances (lab, pitch -5 deg: length
   within about 0.5 mm of the tape measure). Then set `wrist2_branch_1g` [T8] to the arm configuration used.

Save as e.g. `eoat_kiro_02.yaml` and pass `calib_file:=`.

## 6. Action interface

`/hongik_fine_sensing/precision_sensing` (`welding_interfaces/action/PrecisionSensing`):

| | |
|---|---|
| Goal | `welding_position` 1 (1G; any other value -> `ERROR_INVALID_GOAL`), `pose_a`, `pose_b` (positions used) |
| Feedback | `elapsed_time` [s] since acceptance, 1 Hz and on every progress line; the lines themselves: topic `/hongik_fine_sensing/hongik_auto_teach_server/log` (`std_msgs/String`, reliable, one line per message) |
| Result | `error_code`, `error_message` = `"<status>: <reason>"`, `paths` (one `WeldingPath` per pass of the weld plan on success - 4 for the 10T groove), `data_file_path` (absolute path of `precision_sensing.npz` on this PC; empty when nothing was measured) |
| Cancel | stops our trajectory at once (decelerating stop on the JTC) and ends the run; result `canceled`, code 255, message `ABORTED: cancel requested ...` |
| Result file | the full result message (every field) in `<run folder>/precision_sensing_result.yaml`, written before the result is returned for every goal whose run made a folder; progress line `precision_sensing: result file <path>` |
| Base frame | `~/status` (`std_srvs/Trigger`) names it: `base_frame=World` |

| Our end | `error_code` |
|---|---|
| `PASS`, `PASS_UNRESOLVED` (budget used; the message lists the quantities still outside k sigma <= tol) | 0 SUCCESS + paths |
| invalid input (position, points, job file) | 5 ERROR_INVALID_GOAL |
| planning refused (MoveIt, envelope), trajectory rejected / aborted by the JTC, platform not ready | 1 ERROR_MOTION_FAILED |
| sensor connection / start | 2 ERROR_SENSOR_FAILED |
| seam not recognised (no groove, ends not identified) | 3 ERROR_SENSING_FAILED |
| weld plan not built | 4 ERROR_WAYPOINT_GENERATION_FAILED |
| configuration, compute errors, busy, cancel | 255 ERROR_INTERNAL |

`/hongik_fine_sensing/post_weld_scan` (`hongik_scan_msgs/action/PostWeldScan`; terminal client `ros2 run hongik_scan_ros
hongik_post_weld_scan [--run <run folder>] [--after-pass K] [--measure-speed V] [--dry-run]`): only when called, after the welding or
after any pass (`after_pass`; the caller decides when) - one straight scan of the whole weld along the measured seam of a finished 1G
run (`run_dir`, '' = the latest under `out_root`), 20 mm beyond each measured end, at that run's pass standoff and sensor orientation
and its measurement speed, in that run's world (MoveIt scene and envelope). `measure_speed_mps` / `--measure-speed V` [mm/s] chooses
another speed for the scan line: any speed inside the calibration's noise-prior speed range (`g1.noise_priors.speed_range_mps`,
15..60 mm/s for the shipped priors) - the same single check as a run's measurement speed chosen at run time; outside it the goal ends
REFUSED before any motion (nothing sent). Raw data only: nothing evaluates the profiles, the weld plan is not
changed; output `<run_dir>/post_weld_scan_p<k>|final_<stamp>/` = `raw.npz` + `run_log.json` + the sensor parameter backup. Result
`status` OK / DRY_RUN / REFUSED / ABORTED / ERROR with a one-line `message`; one goal at a time over all actions.

## 7. Outputs

`WeldingPath.waypoints` per pass (welding order) = the pass's straight line in WeldingPath's documented form: position = the welding
wire tip = the bead aim point (the wire end), base frame, m, evenly <= 5 mm apart; +Z = the pass's plate normal on the torch side, +X =
along the line, +Y = Z x X; the last waypoint repeats the previous orientation. `kiro/pass_<k>.yaml` gives the same line as TCP poses
(`right_manipulator_ee_point`, the contact-tip end centre) = aim point - (nozzle end + stick-out) x z_TCP (meta
`kiro_outputs.tcp_to_aim_point_mm`; KIRO's executor does not apply a stick-out). The TCP orientation (torch work / travel angles, the
TCP roll chosen by D14), the World poses and the cap weave are in `kiro/` and the npz.

Run folder `<out_root>/<YYYYmmdd-HHMMSS>_kiro_1G/`:

| File | Content |
|---|---|
| `precision_sensing_result.yaml` | the action result message as returned (error_code, error_message, paths with every waypoint, data_file_path) |
| `precision_sensing.npz` | `meta_json` (status, codes, frames, inputs, start pose at acceptance, `ends_from_goal`, weld direction), `sec_*` (cross sections), `n_paths`, per pass `p{k}_wp_pos_m`, `p{k}_wp_quat_xyzw` (= the result paths), `p{k}_s_mm`, `p{k}_R_torch`, `p{k}_speed_mps`, `p{k}_area_mm2`, `p{k}_weave_*`, `p{k}_info_json`, `p{k}_dense_*` |
| `sections.json` / `.npz` | cross sections along the seam: root, plate tops, bevel / root faces, per-face normals, sigmas, ends, length |
| `weld_plan.json` / `.npz` | the 4-pass plan (`schema_version` 3: convex beads + weaving cap, per pass the aim rule and its reason, the bead section per station - toes, crown, crown ratio -, speeds, dense paths, TCP poses, reachability check D14) |
| `kiro/` | `pass_<k>.yaml` (KIRO's `xy_endpoint_review_v1`: `current_start` -> `current_goal` of pass k, frame `World`, poses of `right_manipulator_ee_point`), `pass_<k>_info.yaml` (the assumed WPS settings and derived layer values - information, not commands), `README.md` |
| `verification/` | raw sensor data, per-leg fits, run log (`run_log.json`) |

S / E naming: S = the seam end the run visited first; `meta.ends_from_goal` says whether that was `pose_a` or `pose_b`.

## 8. Troubleshooting

| Symptom | Check |
|---|---|
| server exits "cannot load the KIRO action type" | `welding_interfaces` not built / registered: INSTALL step 2 (colcon.meta) and `source install/setup.bash` |
| `platform not ready: ...` | the named item: move_group services, JTC action, `/joint_states` with the chain joints of `/robot_description` (FK pose), the URDF on `/robot_description` equal to the robot's TF |
| `platform config: ... is required` / `not a setting of this package` | `hongik_kiro.yaml` keys (section 3) |
| `REFUSED ... inside the cell world` | move the arm clear of the specimen / support before the goal |
| `REFUSED ... refused before motion by the envelope (D3)` / `by the motion planner (D4)` | the approach is too close to the plate or not reachable: start pose, points, `[T6]` tool geometry |
| `sensor connection/initialisation failed` | sensor LAN (`ping`), only one client at a time (stop other tools), `sensor.ip` |
| `DATA_FAIL: groove not in view at the ... rough point` | points on the seam inside the specimen span (section 4), specimen design in the job file |
| progress | `ros2 topic echo /hongik_fine_sensing/hongik_auto_teach_server/log`; state: `ros2 service call /hongik_fine_sensing/hongik_auto_teach_server/status std_srvs/srv/Trigger` |
