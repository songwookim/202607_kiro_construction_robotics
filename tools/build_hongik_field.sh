#!/usr/bin/env bash
# Build the RELAY field source without touching the existing robot workspace outputs.
set -euo pipefail
task_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_workspace="$(cd -- "$task_project_root/../.." && pwd)"
task_source="$task_project_root/hongik_fine_sensing_field"
task_overlay="${1:-$HOME/hongik_field_ws}"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "Deactivate the Python venv before building the field overlay." >&2
  exit 2
fi
set +u  # ROS setup scripts read optional unset variables.
source /opt/ros/humble/setup.bash
source "$task_workspace/install/setup.bash"
set -u
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/usr/lib/python3/dist-packages"
exec /usr/bin/colcon --log-base "$task_overlay/log" build \
  --base-paths "$task_source/scan_core" "$task_source/scan_l0" \
    "$task_source/vc3dz_driver" "$task_source/hongik_scan_ros" \
    "$task_source/hongik_scan_msgs" "$task_source/eoat_description" \
  --build-base "$task_overlay/build" --install-base "$task_overlay/install" \
  --packages-select scan_core scan_l0 vc3dz_driver hongik_scan_ros hongik_scan_msgs eoat_description \
  --allow-overriding scan_core scan_l0 vc3dz_driver hongik_scan_ros hongik_scan_msgs eoat_description \
  --symlink-install --parallel-workers 2
