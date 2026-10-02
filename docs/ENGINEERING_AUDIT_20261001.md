# Engineering audit — 2026-10-01

Scope: current Tk production implementation, application/core/nodes/io, ROS
configuration, and non-Qt tests. No physical stack was launched, and no robot,
welder, or Fastech commands were sent. Existing keyboard ramp/profile changes
and their tests were preserved without editing them.

## Findings and implemented fixes

| Class | Evidence / problem | Change and significance |
| --- | --- | --- |
| A/B | `WeldActionGui.stop_sequence()` finalized feedback (analysis, file writes, plot startup) before robot cancellation. | Send existing cancel/stop requests first; detach the logging session immediately and queue UI-triggered saves on a single application-owned writer. STOP no longer waits for log analysis/disk work. Captured end timestamps exclude writer queue time. |
| A | `WeldGuiNode._send_action_goal_and_wait()` could time out before acceptance, then a late callback claimed the active goal without cancellation. Cartesian execution also used one shared inner MoveIt handle. | `nodes/action_call.py` retains ownership of delayed responses, cancels late accepted goals, and prevents submission after cancellation during discovery. Runtime tracks pending calls; Cartesian cancellation is scoped to the parent goal UUID, including cancellation before inner registration. |
| A | An exception in a queued Tk callback aborted the drain and prevented scheduling the next ROS/UI bridge tick. | Isolate/report callback exceptions and reschedule the bridge in `finally`; subsequent UI updates are not discarded. |
| A | Fastech hardware access was locked, but publishing the resulting snapshot was not. A poll could publish old state after a newer output readback/disconnect. | Serialize read/command/connect/disconnect **and publication** as one node transaction. Adapter, mapping, polling cadence, and public interfaces remain unchanged. |
| B | Cartesian service/action waits checked completion every 10 ms. | Wake on Future completion using an Event; remove per-response polling delay without changing timeout constants or trajectory calculation. |
| C | Feedback controller wrappers were recreated per call, unsuitable for ownership of background saves. | Cache one recorder per GUI; keep serialized save/cleanup ownership in `application/`, with synchronous Path returns preserved for worker callers. |

Action cancellation remains cooperative: a late accepted goal is canceled as
soon as its handle is available. This does not guarantee that a remote robot
cannot begin moving before acknowledgment, nor replace controller STOP.

## Measurement

Software-only microbenchmark: 100 sequential futures completed by a
`threading.Timer(0.001)`, measuring timer submission to wait completion.

| Waiting method | Median | Maximum |
| --- | ---: | ---: |
| Previous 10 ms polling | 10.401 ms | 20.442 ms |
| Completion Event | 1.299 ms | 13.130 ms |

This measures Python wake-up behavior, not DDS round-trip or physical motion
latency. STOP latency reduction is structurally justified by removing file
processing from its critical path; no hardware timing improvement is claimed.

## Intentionally unchanged / follow-up

- Keyboard command state is already integrated locally by the Jacobian jog
  node; JTC stays active. Focus/deadman authorization still passes through Tk.
  Bypassing that path changes safety ownership, so no new listener, controller
  switch, ramp, limit, watchdog, or lookahead tuning was introduced.
- The legacy direct asynchronous GUI action paths (`_cartesian_goal_response`,
  `_initial_execute_goal_response`) still use the shared active-handle model.
  They need a separate lifecycle migration with their UI/touch-guard tests;
  this patch fixes the synchronous workflow helper and Cartesian server, not
  every possible action-submission path.
- Application controllers still reference some GUI host state. Wholesale
  extraction, workflow shutdown/join redesign, and direct hardware ownership
  changes would exceed a narrow behavior-preserving fix.
- A service timeout cannot retract an already sent remote output request.
  Command expiry/acknowledgment protocols require an explicit interface design;
  they were not invented during cleanup.
- Hi-COMM cadence, telemetry coalescing, packet fields, ARC ordering, native/
  software crater, seam/weave mathematics, multipass cumulative state and
  provenance checks were preserved. No unmeasured high-rate tuning or cosmetic
  class/file splitting was performed. Class D changes were not pursued.

## Validation

- Baseline non-Qt suite: **328 passed** (5.97 s).
- Focused modified-area tests: **158 passed** (2.80 s); final focused lifecycle,
  UI, recorder and Fastech rerun: **30 passed** (1.37 s).
- Full non-Qt suite after fixes: **341 passed** (6.32 s). Tests cover late action
  acceptance/result timeout, pre-submission STOP, parent-scoped cancel, UI
  exception isolation, STOP ordering, serialized Fastech publication and
  background save timestamp/session ownership.
- All **7** production console entry points imported; their `main()` functions
  were not invoked. Qt was neither modified nor tested. The hardware exercise
  script `test/rbpodo_test.py` was excluded explicitly.
- `git diff --check` passed.
- Colcon: **4 packages built successfully** (`construct_robot`,
  `construct_msgs`, `construct_moveit_config`, `construct_description`, 19.3 s).
  Isolated output: `/tmp/construct-audit-build-GreiKu/{build,install,log}`;
  workspace `build/`, `install/`, `log/` were not overwritten. Two CMake packages
  warned that `PYTHON_EXECUTABLE` was unused; no build failure.

Test invocation (after sourcing Humble and the existing workspace):

```bash
export PYTHONPATH="$PWD/construct_robot:$PYTHONPATH"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q construct_robot/test \
  --ignore=construct_robot/test/rbpodo_test.py
```

Required supervised hardware follow-up: verify STOP during a long log session;
cancel while MoveIt acceptance is delayed; verify touch stop/retract and
Fastech output/readback ordering under polling; measure keyboard press/release
and teaching-transition behavior with the existing user profile changes.
Unit tests and successful builds do not establish hardware safety.
