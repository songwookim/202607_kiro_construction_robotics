#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <tesseract-overlay-src-directory>" >&2
  exit 2
fi

overlay_src="$1"
processor="$(uname -m)"
if [[ "${processor}" != "aarch64" && "${processor}" != "arm64" ]]; then
  echo "No ARM64 patch required on ${processor}"
  exit 0
fi

macro_files=(
  "${overlay_src}/descartes_light/descartes_light/core/cmake/core-macros.cmake"
  "${overlay_src}/opw_kinematics/cmake/opw_kinematics_macros.cmake"
  "${overlay_src}/tesseract/tesseract_common/cmake/tesseract_macros.cmake"
  "${overlay_src}/trajopt/trajopt_common/cmake/trajopt_macros.cmake"
)

for macro_file in "${macro_files[@]}"; do
  sed -i \
    '/^[[:space:]]*-mno-avx[[:space:]]*$/d; /^[[:space:]]*set([^)]* -mno-avx)[[:space:]]*$/s/ -mno-avx//' \
    "${macro_file}"
done

sco_cmake="${overlay_src}/trajopt/trajopt_sco/CMakeLists.txt"
sed -i \
  's/^if(NOT APPLE AND NOT WIN32)$/if(NOT APPLE AND NOT WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86")/' \
  "${sco_cmake}"

echo "Applied Tesseract 0.22 ARM64 compiler and BPMPD compatibility patches"
