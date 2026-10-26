# 1G / 3G welding tasks

`2. TASK` provides `Welding · 1G` and `Welding · 3G`. Both use the existing
teaching, planning, welding, and four-pass controls. Torch Cleaner and Scan
remain shared equipment pages. The active welding task is shown above the
task tabs, including when the operator opens a shared page.

Each welding task keeps its own teaching, seam correction, four-pass state,
sequence, path, and weld parameters. Changing tasks requires idle equipment
and invalidates the previous plan approval. Changing tasks never starts motion
or sends welding commands. A new 3G task starts without taught poses or pass
references; teach the poses and set the process conditions for the actual job.

| Data | 1G | 3G |
| --- | --- | --- |
| Teaching poses and seam/touch YAML | `construct_description/config/` | `construct_description/config/welding_3g/` |
| Task settings | `construct_description/config/task_settings.yaml` | `construct_description/config/welding_3g/task_settings.yaml` |
| Welding feedback and plots | `~/ros2_ws/weld_feedback/` | `~/ros2_ws/weld_feedback/3g/` |

Existing 1G files retain their paths. Each feedback folder has its own
`latest_weld_feedback.log`; new feedback records include `welding_position`.
The recording task and output directory are frozen when recording starts,
including when the log is serialized in the background.

Captured teaching poses are saved immediately through the existing YAML
workflow. `Save task settings` saves parameter edits without welding. Settings
are also saved when changing welding tasks and when closing the GUI, and
restored on the next use. Generated sequences and unsaved correction working
state are retained across task switches within the running GUI; the existing
pass/reference persistence workflow continues to handle saved pass files.

The existing `Scan + 4 Passes` workflow uses 1G sensing and is available only
with the 1G task selected. This change does not add 3G support to the sensing
server or derive a 3G process recipe. A combined work cycle on 3G requires its
own `welding_3g/combined_work_cycle.yaml` configuration.

## Validation

Run from `src/construct_robot_ros2` with ROS 2 Humble available:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/construct_robot:$PYTHONPATH" /usr/bin/python3 -m pytest -q -p no:cacheprovider \
  construct_robot/test/test_welding_tasks.py \
  construct_robot/test/test_weld_execution_controller.py \
  construct_robot/test/test_scan_four_pass.py \
  construct_robot/test/test_multipass_controller.py \
  construct_robot/test/test_seam_correction_controller.py \
  construct_robot/test/test_four_pass_correction.py \
  construct_robot/test/test_scan_teaching.py \
  construct_robot/test/test_scan_weld_cycle.py \
  construct_robot/test/test_gui_wheel_weave.py \
  construct_robot/test/test_sequence_executor.py
```

These tests exercise state/storage isolation, teaching paths, settings reload,
busy-operation guards, actual Tk frame mounting, late IK callbacks, recording
directory capture, and the adjacent scan, correction, and sequence workflows.
They do not connect to the physical robot or welder. The Tk frame test skips
when no display is available.
