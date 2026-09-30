# construct_robot Refactoring Guide for Claude

## Purpose

This document describes the intended architecture and refactoring constraints for the current ROS 2 Python repository:

`202607_kiro_construction_robotics/construct_robot`

Use this as architectural context before modifying the repository.

The goal is **not to rewrite the system**. The goal is to make the current implementation easier to understand, test, extend, and maintain while preserving production behavior.

The production GUI is **Tkinter**. PySide6 work is not the current production direction.

---

## Source of truth

Use the **current source code and tests** as the source of truth.

Do not infer behavior from stale documentation when it conflicts with the implementation. Unit tests are not a substitute for real-hardware validation.

---

## Current package direction

The repository has already been reorganized into role-oriented packages:

```text
construct_robot/
└── construct_robot/
    ├── nodes/
    │   ├── cartesian_path_server.py
    │   ├── controller_spawner.py
    │   ├── fastech_io_node.py
    │   └── keyboard_teaching_node.py
    │
    ├── gui/
    │   ├── weld_action_gui.py
    │   ├── task_teaching_panel.py
    │   └── torch_cleaner_panel.py
    │
    ├── core/
    │   ├── cartesian_path_common.py
    │   ├── multipass.py
    │   ├── seam_geometry.py
    │   ├── sequence_model.py
    │   ├── task_teaching_model.py
    │   ├── torch_cleaner_teaching.py
    │   ├── weld_config.py
    │   ├── weld_quality_metrics.py
    │   └── work_cycle.py
    │
    ├── io/
    │   ├── fastech_ethernet.py
    │   ├── hicomm_welder.py
    │   ├── teaching_yaml.py
    │   └── weld_logging.py
    │
    └── gui_qt/
        └── ...
```

Some top-level compatibility modules remain for old imports. Treat them as migration shims unless proven unnecessary.

---

# Main architectural problem

`gui/weld_action_gui.py` is still much too large and owns too many responsibilities.

At the latest reviewed state it is approximately:

- 17.6k lines
- ~753 KB
- `WeldGuiNode`: ~88 methods
- `WeldActionGui`: ~319 methods

It currently mixes:

- Tkinter widget construction and view state
- ROS 2 clients, publishers, subscribers, TF and action handling
- MoveIt integration
- robot controller switching and robot power
- Fastech runtime integration
- Hi-COMM welding execution
- weld feedback recording
- keyboard teaching control
- seam probing and seam correction workflow
- multi-pass workflow
- weld scenario generation
- sequence editing and sequence execution
- YAML/file persistence
- background threading
- safety synchronization

The desired end state is that `weld_action_gui.py` becomes primarily a **Tkinter composition and view layer**.

---

# Target architecture

Preferred dependency direction:

```text
GUI
 ↓
Application / Controllers
 ↓
Core
 ↘
  Nodes / IO adapters
```

A reasonable target structure is:

```text
construct_robot/
└── construct_robot/
    ├── gui/
    │   ├── weld_action_gui.py
    │   ├── connection_panel.py
    │   ├── welding_panel.py
    │   ├── seam_correction_panel.py
    │   ├── sequence_panel.py
    │   └── torch_cleaner_panel.py
    │
    ├── application/
    │   ├── weld_sequence_builder.py
    │   ├── sequence_executor.py
    │   ├── weld_execution_controller.py
    │   ├── seam_correction_controller.py
    │   ├── multipass_controller.py
    │   └── keyboard_teaching_controller.py
    │
    ├── nodes/
    │   ├── weld_runtime_node.py
    │   ├── cartesian_path_server.py
    │   ├── controller_spawner.py
    │   ├── fastech_io_node.py
    │   └── keyboard_teaching_node.py
    │
    ├── core/
    │   ├── seam_geometry.py
    │   ├── multipass.py
    │   ├── sequence_model.py
    │   ├── weld_config.py
    │   └── ...
    │
    └── io/
        ├── hicomm_welder.py
        ├── fastech_ethernet.py
        ├── teaching_yaml.py
        ├── weld_logging.py
        └── ...
```

Do not force this exact file list if a simpler cohesive design is better. The important part is the responsibility boundary.

---

# Responsibility rules

## `gui/`

GUI code should own:

- Tk root/window
- Notebook / frames / panels
- buttons, labels, Treeview, dialogs
- Tk variables
- user input collection
- rendering status/results
- view-only formatting
- callback glue from UI to application/controller objects

GUI code should **not** own:

- geometry algorithms
- scenario generation
- welding protocol encoding
- ROS service/action orchestration
- sequence execution engine
- YAML serialization logic
- persistent storage logic
- robot controller switching logic
- ARC synchronization logic
- touch-probe workflow state machines
- multipass registration algorithms
- general background-worker infrastructure

Tk variables such as `StringVar`, `DoubleVar`, and `BooleanVar` are acceptable in the GUI layer.

## `application/`

Use this layer for orchestration that is neither GUI nor pure domain logic.

Examples:

- build a weld scenario from teaching + seam + recipe settings
- run a sequence
- coordinate ARC ON / motion / ARC OFF
- coordinate touch probing stages
- coordinate multi-pass registration flow
- coordinate keyboard teaching workflow
- transform GUI snapshots into application input objects

This layer may depend on `core`, runtime interfaces, and IO abstractions. It should not depend on Tkinter widgets.

## `core/`

Core should contain:

- pure calculations
- domain validation
- state models
- geometry
- immutable/copyable scenario data
- sequence state/model operations

Core should avoid:

- Tkinter
- direct ROS Node ownership
- direct filesystem access where possible
- direct device I/O

Existing good examples include `SequenceModel`, `MultiPassState`, and seam geometry helpers.

## `nodes/`

Nodes should contain actual ROS 2 runtime ownership:

- publishers/subscribers
- action clients
- service clients
- TF
- controller-state queries
- ROS timers
- ROS parameters

Do **not** turn pure Python logic into ROS nodes merely to split files.

## `io/`

I/O should contain:

- external hardware protocol adapters
- file serialization/persistence
- YAML load/save
- log storage
- Hi-COMM framing/transport
- Fastech Ethernet adapter

---

# Immediate high-value extraction targets

## 1. `WeldGuiNode`

Current location: `gui/weld_action_gui.py`

It is effectively a ROS runtime object, not GUI presentation code. It owns ActionClient, ServiceClient, publishers/subscribers, TF, MoveIt/FK/IK, controller switching, robot power, Fastech ROS interfaces, guarded motion, and trajectory execution.

Preferred first move:

```text
gui/weld_action_gui.py
        ↓
nodes/weld_runtime_node.py
```

At first, preserve the existing `ui` callback relationship if necessary. Do not combine moving the class with a major event-system redesign in the same step.

## 2. `build_sensed_weld_sequence()`

This is one of the largest and highest-value extraction targets.

It currently mixes GUI input reads, teaching state, seam state, approach settings, weld settings, weave settings, validation, sequence-step generation, and `SequenceModel` updates.

Preferred direction:

```text
Tk GUI
  ↓
collect WeldScenarioInput
  ↓
application/weld_sequence_builder.py
  ↓
build_weld_sequence(...)
  ↓
list[sequence steps]
  ↓
SequenceModel
```

The sequence builder must not read Tk widgets directly.

## 3. Sequence execution

GUI-owned execution concepts currently include `run_sequence`, `_sequence_worker_body`, `_run_sequence_step`, STOP handling, execution conditions, and slot/parallel coordination.

Preferred direction:

```text
gui/sequence_panel.py
        ↓
SequenceModel
        ↓
application/sequence_executor.py
        ↓
runtime node / IO adapters
```

The GUI renders state. The executor runs the work.

## 4. Seam correction workflow

Pure geometry is already partly in `core/seam_geometry.py`, but workflow orchestration remains in the GUI.

Move orchestration such as automatic touch stages, touch progression, seam correction progression, approach/return sequencing, completion state, and error/result propagation to something like:

`application/seam_correction_controller.py`

Do not move pure geometry back into application code.

## 5. Multi-pass workflow

`core/multipass.py` already contains GUI-independent math/state. Move remaining workflow orchestration toward `application/multipass_controller.py`.

The following semantics must not change:

- Pass 1 updates passes 1–4
- Pass 2 preserves pass 1, updates 2–4
- Pass 3 preserves 1–2, updates 3–4
- Pass 4 updates only 4
- cumulative CURRENT state
- START and GOAL anchors remain separate
- direction change rotates later offsets/orientations
- >30° direction change remains rejected
- source references remain immutable
- provenance/hash validation remains

## 6. File/YAML persistence

`weld_action_gui.py` still performs too much direct YAML/file handling.

Move cohesive persistence responsibilities into `io/`, including seam teaching references, seam touch persistence, multi-pass state persistence, execution settings persistence, and work-cycle YAML loading.

Prefer:

```text
core/
    validate / assemble / model

io/
    load / save / atomic write
```

---

# Production GUI direction

Tkinter is the current production GUI.

Do not redesign the project around PySide6. Do not delete `gui_qt/` as part of an unrelated refactor unless explicitly requested.

The intended Tk operator structure is:

```text
1. CONNECTION
2. TASK
   ├── Welding
   └── Torch Cleaner
3. SEQUENCE BUILDER
```

The product workflow is:

```text
TASK → TEACH → AUTOMATE → VERIFY → EXECUTE
```

Avoid unnecessary deep nesting.

---

# Safety invariants

These must remain unchanged.

## Passive UI must never cause motion/output

The following must not cause robot motion, welding output, or external output by themselves:

- opening the GUI
- switching tabs
- selecting teaching items
- editing parameters
- loading data
- viewing seam correction
- computing seam geometry
- planning/previewing
- viewing sequence rows

## Plan must remain output-free

Plan/preview must not:

- ARC ON/OFF
- send Hi-COMM welding output
- toggle Fastech outputs
- enable torch-cleaner outputs
- physically execute robot motion

## Execute guards

Preserve existing connection checks, operator confirmations, ARC guards, controller-state checks, teaching validity checks, touch safety checks, and STOP behavior.

## STOP behavior

STOP is safety-critical. Do not simplify or redesign it without explicit instruction and hardware validation.

## ARC synchronization

Do not change the semantics of ARC establishment, motion synchronization, ARC OFF timing, triggered ARC OFF, crater sequencing, or hot-start sequencing.

Do not replace event synchronization with a cleaner abstraction unless equivalence is proven.

## Touch probing

Preserve guarded motion, touch stop/cancel, controller restoration, return behavior, capture semantics, and touch-release handling.

## Keyboard teaching

Preserve one authoritative keyboard-teaching path, deadman behavior, focus handling, controller switching, release/zero-command behavior, wire F/R behavior, and existing save shortcuts.

Do not create competing global keyboard listeners.

---

# Threading guidance

The current code has many direct `threading.Thread(...)` uses.

Do **not** globally replace them.

Keep explicit thread/event synchronization for safety-critical workflows such as touch-guarded motion, welding synchronization, ARC handling, sequence execution, and motion completion.

For simple background work only, a common bounded worker abstraction may later be introduced for tasks such as connect/disconnect requests, lightweight persistence, ordinary background calculations, and non-safety service helper jobs.

Avoid introducing `asyncio` unless explicitly requested.

---

# ROS executor guidance

The project uses `MultiThreadedExecutor`.

Do not change callback-group behavior casually. Changing callback groups can alter timing/concurrency for joint-state processing, touch events, keyboard velocity timers, service callbacks, and action callbacks.

Any callback-group refactor should be isolated and followed by focused tests plus hardware validation.

---

# Configuration cleanup

There are duplicated configuration values across ROS parameters and Tk state, including robot IPs, Hi-COMM IPs, and Fastech IP.

Long-term direction:

```text
ROS parameters / runtime configuration
          ↓
authoritative config object
          ↓
GUI display/edit layer
```

Do not allow duplicated defaults to drift independently.

---

# Dependency direction issues

Watch for dependencies such as:

```text
core → io
```

For example, protocol validation may currently reach from core into Hi-COMM I/O implementation.

A possible future direction is:

```text
core ─────┐
          ↓
      protocol
          ↑
io ───────┘
```

Do not perform this merely for architectural purity. Only do it when it materially improves testability or ownership while preserving behavior.

---

# Compatibility shims

Top-level old modules may remain temporarily to preserve imports.

Do not delete compatibility shims until all repository imports use the new paths, tests no longer require them, entry points do not require them, saved/legacy tooling does not require them, and removal is confirmed safe.

Compatibility cleanup should be a separate final phase.

---

# Naming guidance

Prefer clear role-based names such as:

```text
*_node.py
*_panel.py
*_controller.py
*_model.py
*_geometry.py
*_builder.py
*_executor.py
*_store.py
*_protocol.py
```

Do not rename public symbols solely for cosmetic consistency.

---

# Refactoring style

Use:

> move, extract, preserve

Avoid:

> rewrite, simplify, redesign

Safe process:

```text
1. identify one cohesive responsibility
2. extract it with minimum semantic change
3. update imports
4. run focused tests
5. run relevant integration tests
6. continue
```

Do not combine unrelated cleanup with a safety-sensitive extraction.

---

# Suggested phases

## Phase 1 — Low risk

- move `WeldGuiNode` to `nodes/weld_runtime_node.py`
- move pure persistence helpers to `io/`
- extract UI-independent weld scenario generation
- preserve compatibility imports if needed
- do not change runtime semantics

## Phase 2 — Medium risk

- extract sequence execution
- reduce sequence editor duplication
- extract application-level controllers
- consolidate runtime configuration
- remove GUI access from application logic

## Phase 3 — Higher risk

- seam correction workflow extraction
- multi-pass workflow extraction
- keyboard teaching controller extraction
- runtime callback/event interface cleanup

## Phase 4 — Final cleanup

- remove verified dead code
- remove no-longer-needed compatibility shims
- evaluate whether `gui_qt/` should remain
- reduce remaining direct threading boilerplate
- evaluate ROS callback groups only with hardware validation

---

# Validation requirements

After each cohesive extraction:

- run focused unit tests
- run import/entry-point tests
- verify `colcon build`
- keep existing ROS executable names working
- do not change launch semantics unintentionally

Before declaring the refactor production-ready, manually validate on hardware:

- Plan produces zero external outputs
- Execute connection guards
- ARC confirmation/guards
- STOP
- keyboard deadman
- touch stop/return
- multi-pass START/GOAL capture
- cleaner DO outputs
- Hi-COMM communication
- Fastech communication
- weld feedback recording

Unit tests alone are not hardware validation.

---

# Success criteria

Do not judge success only by line count.

A successful refactor means:

```text
gui/weld_action_gui.py
    = Tk view + callback glue

application/
    = workflow orchestration

core/
    = calculation + state

nodes/
    = ROS runtime

io/
    = hardware/file boundaries
```

The best sign of success is that application/core modules can be tested without creating a Tk root and without starting ROS nodes.

A smaller `weld_action_gui.py` should follow naturally, but line-count reduction is secondary to responsibility separation.

---

# Stop conditions

If a refactor step becomes unsafe or too broad:

- finish the current coherent extraction
- keep the repository buildable
- keep tests passing
- stop before starting the next risky boundary

Never leave the package with half-migrated imports or partially moved runtime ownership.

A smaller completed refactor is better than a larger incomplete rewrite.
