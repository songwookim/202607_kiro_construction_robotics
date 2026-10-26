"""Where scan data and scan teaching results live.

Raw scan recordings (scan.npz, scan.json, hongik_estimate/): ~/ros2_ws/scan_data/kiro_scan/<time>_scan/
(override: KIRO_SCAN_ROOT). Accumulated teaching results (reference, correction, per-scan estimate history):
construct_description/config/scan_teaching/.
"""

import os
from pathlib import Path


def scan_results_root():
    return Path(os.environ.get("KIRO_SCAN_ROOT", Path.home() / "ros2_ws" / "scan_data" / "kiro_scan")).expanduser()


def post_weld_scan_root():
    """Post-weld scans (recording only, no estimate): ~/ros2_ws/scan_data/post_weld_scan (override: KIRO_POST_WELD_SCAN_ROOT)."""
    return Path(os.environ.get("KIRO_POST_WELD_SCAN_ROOT",
                               Path.home() / "ros2_ws" / "scan_data" / "post_weld_scan")).expanduser()


def ensure_scan_results_root():
    """Create the root; keep colcon and git out of it (it sits inside the workspace)."""
    root = scan_results_root()
    root.mkdir(parents=True, exist_ok=True)
    data = root.parent
    for marker, text in (("COLCON_IGNORE", ""), (".gitignore", "*\n")):
        path = data / marker
        if not path.exists():
            path.write_text(text)
    return root
