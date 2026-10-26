# Pass-2 anchored position calibration

P2 is (lateral toward the 90-degree face, along, up) = (2, 0, 5) mm from the P1 root wire tip. Root = measured P2 wire tip minus this design offset, separately at both ends. START uses the saved 2026-10-15 14:43:43 reference; GOAL uses the existing 01:55:12 reference.

This is a position correction for the same fixture, using the existing scanner hand-eye and wire-tip calibration. Other saved pass XYZ are ignored; pass torch orientations are retained. P3 and P4 keep their existing design offsets. No robot motion or arc was executed.

Source scan: `/home/irs/ros2_ws/scan_data/kiro_scan/20261015-140509_scan` (unchanged).

Restart weld_action_gui to load the new code and calibration. In Scan, use Position anchor pass 2, Apply correction checked and design_layout. For this exported teaching, select this folder in Welding > 4-PASS, Load 4 references, then select/apply a pass. Rebuild any previously built Scan + 4 Passes sequence so its settings reflect the new calibration.

Generated paths require IK and the existing planning/clearance checks. calibration_report.json records positions, algebraic reproduction error and unchanged-source hashes. This reproduction check does not establish physical accuracy.

Previous configuration backup: `/home/irs/ros2_ws/src/construct_robot_ros2/construct_description/config/scan_teaching/history/20261015-145751_before_pass2_origin_calibration`.
