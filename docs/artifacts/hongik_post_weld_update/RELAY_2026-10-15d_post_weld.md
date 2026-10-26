# Hongik -> KIRO field (2026-10-15 01:5x): post-weld scan from the weld pose; explicit directions

## What changed (patch `kiro-1g-field_postweld_vs_3009cdb.patch`, on top of the start-tolerance patch you applied)

1. The post-weld scan may start where the weld left the arm: when the arm is inside OUR grown objects only (hongik/*), it first moves
   straight up along the measured plate normal, tool orientation unchanged, just far enough that the end pose passes the envelope
   (tool's lowest point >= the clearance, optical origin >= standoff_min). MoveIt checks that line with our objects out of the scene
   for that line only (the robot itself and your fixed bodies stay checked); then our objects go back and the scan proceeds.
   Anything else touching (self-collision, a body that is not ours) is still refused.
2. Direction and tool pose of the post-weld scan = the least TCP motion from the current pose (torch rotation first, then the torch
   tip's distance) - after a weld the torch keeps its side; no more 331 deg turns from the old joint-sum choice.
3. `--direction S->E|E->S` forces the scan direction (S / E = the run's own S / E). Default: the least TCP motion.
   New action field `direction` in hongik_scan_msgs/PostWeldScan -> the workspace must be rebuilt (the usual colcon build does it).

## Apply

```bash
cd <path>/hongik_fine_sensing
git apply -p2 --check kiro-1g-field_postweld_vs_3009cdb.patch && git apply -p2 kiro-1g-field_postweld_vs_3009cdb.patch
# rebuild the whole workspace (same colcon command - hongik_scan_msgs changed), source, restart OUR launch only (one server)
```

## Use

Post-weld scan right after a pass (the arm may stay at the weld end):
```bash
ros2 run hongik_scan_ros hongik_post_weld_scan --run <run folder> --after-pass 1
```
Force a direction if wanted: add `--direction E->S` (or `S->E`). The scan line itself is unchanged: the measured seam
[S - 20, E + 20] mm at the run's pass standoff / pitch / measurement speed, raw data only.

## Weld direction you chose (E->S)

To get E->S as the primary output (`kiro/pass_<k>.yaml` and the action result) set it in the job file or at launch:
`weld_plan.direction: E->S` in job_kiro.yaml, or `job_overrides:='{weld_plan.direction: E->S}'` on our launch.
Both sets stay in `directions/S_to_E/` and `directions/E_to_S/`. S = the end the sensing visited first (`ends_from_goal` in
precision_sensing.npz says whether S is pose_a or pose_b).
