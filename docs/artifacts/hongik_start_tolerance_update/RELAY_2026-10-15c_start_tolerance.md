# Hongik -> KIRO field (2026-10-15 00:3x): run 002753 "checked joint plan start changed: max joint 0.0542 deg"

Our error, not the robot's: the check before sending the checked joint trajectory used a 0.05 deg / 0.1 mm threshold (meant only for
"may an earlier plan be reused") - a 0.054 deg holding difference refused the run. It now uses move_group's own rule,
`trajectory_execution.allowed_start_tolerance` (read once from `/move_group/get_parameters`; MoveIt's documented default 0.01 rad
= 0.57 deg when it is not set). The whole trajectory is still checked against the current scene right before sending.

## Apply (on top of what you have now: the TCP-motion patch already applied = our 287bee5)

```bash
cd <path>/hongik_fine_sensing
git apply -p2 --check kiro-1g-field_starttol_vs_287bee5.patch && git apply -p2 kiro-1g-field_starttol_vs_287bee5.patch
# rebuild (same colcon command), source, restart OUR launch only (one server: ros2 action info ... -> Action servers: 1)
```
A fresh delivered tree takes the cumulative `kiro-1g-field_vs_ec00145.patch` instead.

After the update the start line reads `S = the point visited first (decided before any motion: least TCP (torch tip) motion ...)`;
a refusal of this kind would name the tolerance and its source (`move_group trajectory_execution.allowed_start_tolerance` or
`MoveIt default ...`).

Also still open: `sensor parameters before this session already equal the INIT values` - restore the sensor once from its original
backup (`python -m vc3dz_driver.cli --ip 192.168.3.15 restore --file <backup_<date>.txt>`).
