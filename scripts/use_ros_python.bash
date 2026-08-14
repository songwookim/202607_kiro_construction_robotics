#!/usr/bin/env bash
# Source this file before building or running this ROS 2 workspace.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source this script: source scripts/use_ros_python.bash" >&2
  exit 2
fi

_construct_repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_construct_ws_dir="$(cd "${_construct_repo_dir}/../.." && pwd)"
_construct_venv="${_construct_ws_dir}/.venv"

if [[ ! -x "${_construct_venv}/bin/python" ]]; then
  echo "missing Python environment: ${_construct_venv}" >&2
  return 1
fi

# Remove Conda/Anaconda executables while retaining the rest of the user's
# PATH, then put the workspace Python 3.10 environment first.
_construct_clean_path=""
_construct_old_ifs="${IFS}"
IFS=:
for _construct_path_entry in ${PATH}; do
  case "${_construct_path_entry}" in
    *miniconda*|*anaconda*) continue ;;
  esac
  if [[ -z "${_construct_clean_path}" ]]; then
    _construct_clean_path="${_construct_path_entry}"
  else
    _construct_clean_path="${_construct_clean_path}:${_construct_path_entry}"
  fi
done
IFS="${_construct_old_ifs}"

unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_PROMPT_MODIFIER
unset CONDA_PYTHON_EXE _CE_CONDA _CE_M
export PATH="${_construct_venv}/bin:${_construct_clean_path}"
export VIRTUAL_ENV="${_construct_venv}"
export CONSTRUCT_ROBOT_PYTHON="${_construct_venv}/bin/python"

source /opt/ros/humble/setup.bash
if [[ -f "${_construct_ws_dir}/install/setup.bash" ]]; then
  source "${_construct_ws_dir}/install/setup.bash"
fi

hash -r
echo "ROS Python: $(${CONSTRUCT_ROBOT_PYTHON} -c 'import sys; print(sys.executable, sys.version.split()[0])')"

unset _construct_repo_dir _construct_ws_dir _construct_venv
unset _construct_clean_path _construct_old_ifs _construct_path_entry
