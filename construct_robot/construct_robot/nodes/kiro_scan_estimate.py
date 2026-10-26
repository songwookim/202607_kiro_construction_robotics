"""KIRO seam estimate of a recorded scan (separate process; no ROS, no robot motion).

    python3 -m construct_robot.nodes.kiro_scan_estimate <scan_dir> [--weld-direction E->S] [--calib CALIB]

Writes <scan_dir>/hongik_estimate (estimate.json, kiro/pass_<k>[_info].yaml) and prints one line "RESULT <json>".
"""

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir")
    ap.add_argument("--weld-direction", choices=("E->S", "S->E"), default="E->S")
    ap.add_argument("--calib", default=None)
    a = ap.parse_args(argv)
    from construct_robot.application.kiro_estimate import run_kiro_estimate
    try:
        report = run_kiro_estimate(a.scan_dir, weld_direction=a.weld_direction, calib_path=a.calib,
                                   log=lambda text: print(text, file=sys.stderr, flush=True))
    except Exception as error:                     # GrooveEstimateError carries the operator's reason
        report = {"status": "REFUSED" if type(error).__name__ == "GrooveEstimateError" else "ERROR",
                  "estimator": "kiro", "message": f"{type(error).__name__}: {error}"}
    print("RESULT " + json.dumps(report, default=str), flush=True)
    return 0 if report["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
