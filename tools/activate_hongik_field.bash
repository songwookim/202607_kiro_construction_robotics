# Source this file in a fresh terminal before running Hongik tools or the weld GUI.
task_hongik_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_hongik_workspace="$(cd -- "$task_hongik_project/../.." && pwd)"
task_hongik_overlay="${HONGIK_FIELD_WS:-$HOME/hongik_field_ws}"
if [[ ! -f "$task_hongik_overlay/install/setup.bash" ]]; then
  echo "Hongik field overlay is not built: $task_hongik_overlay/install/setup.bash" >&2
  return 2
fi
source /opt/ros/humble/setup.bash
source "$task_hongik_workspace/install/setup.bash"
source "$task_hongik_overlay/install/setup.bash"
# ROS hooks prepend only paths not already present. If the shell had sourced
# this overlay before, sourcing the underlay above can leave it ahead of us.
# Move this overlay's existing entries to the front without discarding others.
task_hongik_prioritize_path() {
  local task_path_name="$1" task_path_item task_path_joined=""
  local -a task_path_parts task_path_front=() task_path_rest=()
  IFS=: read -r -a task_path_parts <<< "${!task_path_name-}"
  for task_path_item in "${task_path_parts[@]}"; do
    [[ -n "$task_path_item" ]] || continue
    case "$task_path_item" in
      "$task_hongik_overlay"|"$task_hongik_overlay"/*) task_path_front+=("$task_path_item") ;;
      *) task_path_rest+=("$task_path_item") ;;
    esac
  done
  for task_path_item in "${task_path_front[@]}" "${task_path_rest[@]}"; do
    case ":$task_path_joined:" in *":$task_path_item:"*) continue ;; esac
    task_path_joined="${task_path_joined:+$task_path_joined:}$task_path_item"
  done
  printf -v "$task_path_name" '%s' "$task_path_joined"
  export "$task_path_name"
}
for task_hongik_path_var in AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH; do
  task_hongik_prioritize_path "$task_hongik_path_var"
done
unset -f task_hongik_prioritize_path
unset task_hongik_path_var
unset task_hongik_project task_hongik_workspace task_hongik_overlay
