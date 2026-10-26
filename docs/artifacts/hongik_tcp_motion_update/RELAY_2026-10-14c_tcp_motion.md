# Hongik -> KIRO field (2026-10-14 late): sensing posture = least TCP motion

Incremental patch on top of the earlier `kiro-1g-field_vs_ec00145.patch` (already applied on site):
`patches/kiro-1g-field_tcpmotion_vs_70cb923.patch`. A fresh tree can instead take the cumulative
`patches/kiro-1g-field_vs_ec00145.patch` (regenerated, includes this change).

## Apply (KIRO PC)

```bash
cd <path>/hongik_fine_sensing                      # the tree the earlier patch was applied to
git apply -p2 --check <path>/kiro-1g-field_tcpmotion_vs_70cb923.patch && git apply -p2 <path>/kiro-1g-field_tcpmotion_vs_70cb923.patch
# rebuild exactly as before (same colcon command), source again, restart our launch
```
Files: `scan_l0/scan_l0/autoteach.py`, `scan_l0/scan_l0/teach_1g.py` (docstring), `scan_l0/tests/test_sim_teach1g.py`,
`CHANGES_2026-10-14_kiro-1g-field.md` (section 13). No config change, no new parameter.

## What changed

Run 214451 (21:44): both tool orientations of the acquisition were reachable. The old choice took the shortest joint move
(progression -1: 7.3 s, but the torch turned 115 deg around the flange to the side away from the robot base, and the wire had to
follow it). The other orientation (+1) turned the torch only 75 deg and kept it on the base side.

New choice, in this order (no added constraint, no constant):
1. the tool orientation with the smaller rotation of the TCP frame from the current pose (TCP = the torch tip,
   `weld_torch.tcp_frame` = right_manipulator_ee_point, read from the robot model);
2. then the torch tip's straight-line distance (also decides which taught end is visited first);
3. then, for that tool pose, the arm configuration with the shortest planned joint move (as before).
The other orientation is used only when the first one has no usable path. All later passes keep the acquisition's physical tool
pose, so the torch stays on the side chosen here. Practical effect: the torch keeps the side it has at the start pose - start with
the torch on the welder's side.

Run log: `visit_order.options[*].tool_rot_deg / tcp_move_mm`, `visit_order.tcp`, `order_options[*]` show the comparison.

## The other runs of runs.zip (no code change needed)

- 215004: the taught points were 10x too large (`pose_a [0.1711, -18.1688, 8.2180]`) -> rough length 1712 mm -> the 1.8 m specimen box
  in RViz and no reachable acquisition. Check the units of the points given to the action (metres).
- 215221 / 215312 / 215402: refused before motion - the tool at the start pose is 46 mm inside `hongik/specimen` grown by the
  50 mm clearance (`safety.clearance_m`). The large boxes in RViz are these grown clearance volumes (specimen 229 x 305 x 30 mm ->
  329 x 405 x 130 mm). Start with the whole tool more than 50 mm above the specimen top (not at the teaching pose on the plate).

## Send back after the next run

`verification/run_log.json` and the server console (whatever the result).
