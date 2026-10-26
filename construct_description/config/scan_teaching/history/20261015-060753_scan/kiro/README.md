# KIRO-format weld plan (hongik fine sensing)

Written by `scan_core.kiro_plan` from this run's weld plan (`../weld_plan.json`), 4 pass(es). The files under kiro/ give each pass
in the form of KIRO's dataset and list the values we assumed and derived **for sharing - they are not commands**: KIRO's process
settings stay as they are (KIRO's executor keeps its own settings).

## Files
- `pass_<k>.yaml`: KIRO's endpoint format `xy_endpoint_review_v1` (keys, order and types as KIRO's pass_1.yaml ... pass_4.yaml): the straight
  line of that pass's layer, start pose -> goal pose (position + orientation) - KIRO's executor runs it at constant orientation and
  constant TCP speed and generates the weave, lead-in / lead-out and the arc-off pre-distance itself (no dense path, no weave point
  here). `current_start` / `current_goal` = the poses of the TCP `right_manipulator_ee_point` = the torch tip end centre for the usable seam start /
  end of that layer: each pose = the bead aim point of our layer plan (the wire end, on the surface before the pass: the earlier
  passes' convex beads; laterally KIRO's practice from the measured section - pass 1 root: the corner of the root face and the bevel face; pass 2 fill 1: the corner where the root bead meets the square face (the root bead's toe on the square wall); pass 3 fill 2: the lateral centre of the groove at its current surface (midway between fill 1's toes); pass 4 cap: the lateral centre of the groove top (midway between the groove top edges)) at our measured ends
  (the seam ends at the layer height (end-face model, G1WeldPlan.layer_ends)),
  moved back along -z_TCP (the torch axis) by 21.20 mm = the TCP's recess inside the gas nozzle 1.20 mm (derived offline from the URDF's collision geometry (the tool parts the torch axis passes through, farthest along +z_TCP = the gas nozzle rim; right_manipulator_ee_point tool: 37168 triangles (right_manipulator_link6, hongik_eoat_base_link; URDF collision geometry rigid with the TCP: right_manipulator_link6 (right_manipulator_link6_optimized.stl), hongik_eoat_base_link (hongik_eoat_base_link_collision.stl))))
  + the wire stick-out 20 mm measured from the gas nozzle end (job weld_plan.stick_out_mm 20 mm (the wire stick-out from the GAS NOZZLE END (the nozzle rim) to the wire end along the torch axis (welding practice; user 2026-10-13: KIRO's 15-20 mm and our 20 mm are nozzle end -> wire end); provisional 20.0 mm (KIRO's range 15-20 mm; not in KIRO's weld logs - checked 1.log-4.log and pass_1-4.yaml: no stick-out / CTWD, wire_consumable_alpha_mm is a wire-consumption term); replace with KIRO's value. The TCP (the contact-tip end face centre) is recessed inside the nozzle: TCP -> wire end = the torch's nozzle end (weld_torch.nozzle_end_mm, derived from the robot's URDF collision geometry: our torch's CAD 1.2 mm) + stick_out_mm. KIRO's executor does not apply a stick-out (user 2026-10-12 night): our TCP poses are the bead aim points moved back along -z_TCP by that sum)) - KIRO's executor does not apply a
  stick-out. The action result's WeldingPath paths are the aim-point (wire-tip) lines themselves (KIRO's WeldingPath definition).
- `pass_<k>_info.yaml`: information only. `assumed_kiro_settings` = the KIRO settings our layer model assumed (KIRO's own key names
  from its weld logs: `commanded` current_a, voltage, material, diameter_mm, mode, gas; `execution_conditions` weld_tcp_speed_mm_s,
  weld_weave_*, weld_lead_in_mm, weld_lead_out_mm, weld_arc_off_delay_ms). `derived` = what our model derives with them: deposited
  and required areas, layer heights and aim offsets (with their along-seam variation), the cap reinforcement, how far the measured
  seam deviates from the straight line, the travel / work angles of the written orientation and the frames used.

## Frames
- `frame_id: World` (KIRO's schema). World <- World: the runtime planning frame is World (identity) (runtime planning frame).
  Here World is defined as the robot base frame `World` (identity: platform definition) - the poses are in our base frame.
- Poses are of the TCP frame `right_manipulator_ee_point` (the torch tip end centre) of the weld torch model (our 1st-bracket torch (user + KIRO agreement 2026-10-12: the bracket at KIRO is our 1st bracket - same bracket, torch, VC nano 3D-Z and RB20 arm as the lab EOAT; KIRO's URDF 2nd bracket and chiseling gun were never manufactured); nominal TCP = the CAD contact-tip end of our torch (KIRO's TCP = the torch tip end, KIRO 2026-10-12); the plan's tip (bead aim point) = the wire end, the nozzle end (the contact-tip end is recessed inside the gas nozzle) + the process stick-out (job weld_plan.stick_out_mm, from the nozzle end) further along the torch axis; KIRO re-calibrates the TCP on site). Our plan's
  tip frame (origin = the wire end = the bead aim point, z = torch axis toward the work) in that frame (ee <- tip): (+0.00, +0.00, +21.20) mm, axes identity -
  the nozzle end + the stick-out along z_TCP (job weld_plan.stick_out_mm 20 mm (the wire stick-out from the GAS NOZZLE END (the nozzle rim) to the wire end along the torch axis (welding practice; user 2026-10-13: KIRO's 15-20 mm and our 20 mm are nozzle end -> wire end); provisional 20.0 mm (KIRO's range 15-20 mm; not in KIRO's weld logs - checked 1.log-4.log and pass_1-4.yaml: no stick-out / CTWD, wire_consumable_alpha_mm is a wire-consumption term); replace with KIRO's value. The TCP (the contact-tip end face centre) is recessed inside the nozzle: TCP -> wire end = the torch's nozzle end (weld_torch.nozzle_end_mm, derived from the robot's URDF collision geometry: our torch's CAD 1.2 mm) + stick_out_mm. KIRO's executor does not apply a stick-out (user 2026-10-12 night): our TCP poses are the bead aim points moved back along -z_TCP by that sum); axes of our torch's TCP (CAD); the stick-out (nozzle end -> wire end) is a process input). On the flange: `hongik_flange` <- `right_manipulator_ee_point` = (-0.45, +237.86, +297.17) mm, rpy (-0.000, -30.000, -90.000) deg (construct_robot_0528_hongik_v2.urdf r4 (2026-10-13, KIRO's setup identified from its September weld logs): our torch CAD contact-tip end (eoat.xacro torch_tcp, 30.0 deg from the flange axis) turned 180 deg in its clamp about the straight torch body axis (eoat_description/scripts/make_torch_turned_mesh.py), EOAT mounted at eoat_yaw -90 deg (eoat_kiro_01.yaml);
  nominal CAD; KIRO re-calibrates the TCP on site).
- The weld torch is on the arm that made this run (World is its base): weld-plan reachability (D14) is checked on that arm with this torch before a plan is delivered (the result: ../weld_plan.json reachability / feasibility; an offline regeneration does not check it).
- A quaternion q and -q are the same orientation (written with w >= 0).

## Orientation
- Our torch orientation per layer: the process travel and work angles (KIRO dataset 2026-09-15/16, weld_testspecimen_logs; AWS A3.0 definitions: travel = angle out of
  the plane normal to the weld axis, + push / - drag, work = angle from the plate normal in that plane, + toward the bevel face)
  are the target; the plan takes per pass
  the collision-free and reachable orientation closest to them (no bound on the deviation - the deviation is information, listed
  per pass below and in `pass_<k>_info.yaml` derived.orientation), constant over the pass (rotation mean of our section torch
  frames), for KIRO's TCP frame through the ee <- tip transform. The weld-plan check (D14: clearance by tool part role, the robot's
  IK / limits / collisions) ran on exactly these straight lines and orientations (scan_core.weld_line), not on the per-station
  layer plan; `pass_<k>_info.yaml` derived.orientation deviation_why says why a pass left the WPS angles.
  (no weld-pose clearance search for this plan: the WPS angles as planned)
- TCP roll about the torch axis: 180 deg - roll 180 deg: the torch neck bends with the WPS travel angle (NECK_RULE: drag side; the straight body beyond the nozzle on the same side along the travel) - KIRO's executor weaves along the TCP y axis, so the TCP y lies across the seam
  (roll 0 deg: TCP x = travel; 180 deg: TCP x = -travel); the same roll for every pass of this plan.

## Delivery
- These files are written to `kiro/` next to the run's outputs: `<dirname(data_file_path)>/kiro/pass_<k>.yaml` (+ `_info.yaml`,
  README.md) for the PrecisionSensing action (data_file_path = the run's precision_sensing.npz, which embeds the same yaml text as
  kiro_pass_<k>_yaml). The PrecisionSensing action result carries the same straight lines in result.paths, in
  WeldingPath's documented form (robot base frame, <= 5 mm waypoints, every position = the welding wire-tip TCP = our bead aim
  point, +Z = plate normal; first / last waypoint = the aim points of these start / goal poses: TCP position + stick-out x z_TCP,
  in the base frame). WeldingPath has no frame_id and no torch orientation, so the World poses with the torch orientation travel
  in the result's npz (data_file_path: kiro_pass_<k>_start_pose / _goal_pose and the yaml text); see its meta
  `kiro_outputs.action_mapping`.

## Both weld directions
- This run delivers the weld plan in both directions (user decision 2026-10-14); take the set of the direction you weld in:
  - `../directions/S_to_E/kiro/`: every pass S->E, from the S end (the end the sensing run visited first) to the E end
  - `../directions/E_to_S/kiro/`: every pass E->S, from the E end to the S end (the end the sensing run visited first) (the same plan as these files)
  These files (`kiro/`, the action's result.paths and precision_sensing.npz) are the job's direction (weld_plan.direction
  E->S) = the `E_to_S` set.
- Weld every pass of a set in that set's direction and take its pass_<k>.yaml as it is: start / goal and the torch orientation are
  built for that travel (travel angle relative to it, work angle toward the bevel, TCP roll by the torch neck rule) - never swap
  start and goal of a file (the orientation is fixed in space). Per pass either set may be used: pass k of every set lays the same
  layer at the same aim points.

## Notes
- note: offline from KIRO direct scan /home/irs/ros2_ws/scan_data/kiro_scan/20261015-060753_scan
