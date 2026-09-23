"""UI-independent Sequence Builder state and row operations.

Steps retain the existing dictionary schema, including ROS message values.  A
copy is made only where the caller previously made one; execution snapshots
remain the responsibility of the existing sequence runner.
"""

import copy


def next_sequential_slot(steps, requested=1):
    """Return a free slot after every existing non-sleep sequence step."""
    requested = int(requested)
    if not 1 <= requested <= 999:
        raise ValueError("Sequence start slot must be in 1..999")
    occupied = []
    for index, step in enumerate(steps):
        if step.get("type") == "sleep":
            continue
        slot = int(step.get("parallel_slot", index + 1))
        if 1 <= slot <= 999:
            occupied.append(slot)
    return max(requested, max(occupied, default=0) + 1)


class SequenceModel:
    """Canonical ordered rows, independent of Tk selection and rendering."""

    def __init__(self, steps=None):
        self.steps = list(steps) if steps is not None else []
        self.selected_index = None
        self.running = False
        self.current_indices = ()
        self.current_slot = None
        self.group_progress = (0, 0)
        self.status = "idle"

    def replace(self, steps):
        self.steps = list(steps)
        if self.selected_index is not None and self.selected_index >= len(self.steps):
            self.selected_index = None

    def select(self, index):
        self.selected_index = (
            index if index is not None and 0 <= index < len(self.steps) else None
        )
        return self.selected_index

    def execution_snapshot(self, run_all, steps_override=None):
        """Freeze the requested rows without reading GUI selection or widgets."""
        if steps_override is not None:
            indices = list(range(len(steps_override)))
            source = steps_override
        else:
            indices = (list(range(len(self.steps))) if run_all else
                       ([] if self.selected_index is None else [self.selected_index]))
            source = self.steps
        return indices, [copy.deepcopy(source[index]) for index in indices]

    def start(self, indices, execute_requested):
        self.running = True
        self.current_indices = tuple(indices)
        self.current_slot = None
        self.group_progress = (0, 0)
        self.status = "execute" if execute_requested else "plan"

    def set_progress(self, slot, current_group, total_groups):
        self.current_slot = slot
        self.group_progress = (current_group, total_groups)

    def finish(self, success, message):
        self.running = False
        self.current_indices = ()
        self.current_slot = None
        self.status = "complete" if success else f"stopped/failed: {message}"

    def validate(self, require_complete=False):
        return validate_managed_weld_sequence(self.steps, require_complete)

    def add(self, step):
        self.steps.append(step)
        return len(self.steps) - 1

    def extend(self, steps):
        self.steps.extend(steps)

    def edit(self, index, step):
        self.steps[index] = step

    def delete(self, index):
        result = self.steps.pop(index)
        if self.selected_index == index:
            self.selected_index = None
        elif self.selected_index is not None and self.selected_index > index:
            self.selected_index -= 1
        return result

    def clear(self):
        count = len(self.steps)
        self.steps.clear()
        self.selected_index = None
        return count

    def move(self, index, offset):
        target = index + offset
        if not 0 <= target < len(self.steps):
            return None
        self.steps[index], self.steps[target] = self.steps[target], self.steps[index]
        if self.selected_index == index:
            self.selected_index = target
        elif self.selected_index == target:
            self.selected_index = index
        return target

    def duplicate(self, index):
        self.steps.insert(index + 1, copy.deepcopy(self.steps[index]))
        return index + 1

    def with_replaced_cleaner(self, cleaner_steps):
        """Return replacement rows without mutating either input on failure."""
        retained = [step for step in self.steps if not step.get("torch_clean_scenario")]
        next_slot = max((int(step.get("parallel_slot", 0)) for step in retained), default=0)
        if next_slot + len(cleaner_steps) > 999:
            raise ValueError("Cleaner steps exceed the maximum parallel slot 999")
        result = [dict(step) for step in cleaner_steps]
        for offset, step in enumerate(result, 1):
            step["parallel_slot"] = next_slot + offset
        return retained + result


WELD_SCENARIO_STAGE_ORDER = (
    "start_wait", "start_safe", "start_contact", "touch_output_off",
    "arc_on", "custom_hot_start", "weld_motion", "software_crater",
    "arc_off", "goal_wait", "finish", "touch_output_on",
)


def validate_managed_weld_sequence(steps, require_complete=False):
    """Validate generated welding order and its intentional ARC/motion pair."""
    scenarios = {}
    all_slots = {}
    for index, step in enumerate(steps):
        if step.get("type") != "sleep":
            slot = int(step.get("parallel_slot", index + 1))
            all_slots.setdefault(slot, []).append(step)
        scenario_id = step.get("weld_scenario_id")
        if scenario_id is not None:
            scenarios.setdefault(scenario_id, []).append(step)

    motion_types = {"motion", "named_pose", "planned_trajectory"}
    for slot, slot_steps in all_slots.items():
        arc_on = any(
            step.get("type") == "digital_weld"
            and step.get("command") == "on"
            for step in slot_steps
        )
        robot_motion = any(
            step.get("type") in motion_types for step in slot_steps
        )
        if arc_on and robot_motion:
            scenario_ids = {
                step.get("weld_scenario_id") for step in slot_steps
            }
            stages = {
                step.get("weld_scenario_stage") for step in slot_steps
            }
            managed_pair = (
                len(slot_steps) in (2, 3)
                and None not in scenario_ids
                and len(scenario_ids) == 1
                and stages in (
                    {"arc_on", "weld_motion"},
                    {"arc_on", "weld_motion", "arc_off"},
                )
            )
            if not managed_pair:
                raise ValueError(
                    f"D-WELD ON cannot share slot {slot} with arbitrary robot "
                    "motion; only the generated ARC ON + weld-motion + "
                    "triggered ARC OFF group is allowed"
                )

    stage_rank = {
        stage: index for index, stage in enumerate(WELD_SCENARIO_STAGE_ORDER)
    }
    for scenario_steps in scenarios.values():
        previous_rank = -1
        previous_slot = 0
        previous_stage = None
        seen = set()
        for step in scenario_steps:
            stage = step.get("weld_scenario_stage")
            if stage not in stage_rank or stage in seen:
                raise ValueError("Generated weld scenario stages are invalid")
            seen.add(stage)
            rank = stage_rank[stage]
            slot = int(step.get("parallel_slot", 0))
            shared_weld_slot = (
                slot == previous_slot
                and (
                    (stage == "weld_motion" and previous_stage == "arc_on")
                    or (stage == "arc_off" and previous_stage == "weld_motion")
                )
            )
            if rank <= previous_rank or (
                slot <= previous_slot and not shared_weld_slot
            ):
                raise ValueError(
                    "Generated weld scenario order/parallel slots are invalid"
                )
            previous_rank = rank
            previous_slot = slot
            previous_stage = stage

            if stage == "arc_on":
                if (
                    step.get("type") != "digital_weld"
                    or step.get("command") != "on"
                ):
                    raise ValueError("Generated ARC ON stage cannot be changed")
                if float(step.get("duration", 0.0)) != 0.0:
                    raise ValueError(
                        "Generated ARC ON duration must be 0; ARC OFF is explicit"
                    )
                paired_stages = {
                    candidate.get("weld_scenario_stage")
                    for candidate in all_slots.get(slot, ())
                }
                custom_enabled = bool(step.get("settings", {}).get(
                    "custom_hot_start_enabled", False
                ))
                if paired_stages not in (
                    ({"arc_on"},) if custom_enabled else (
                        {"arc_on", "weld_motion"},
                        {"arc_on", "weld_motion", "arc_off"},
                    )
                ):
                    raise ValueError(
                        "Generated ARC ON slot does not match the configured "
                        "custom-hot-start sequence"
                    )
            elif stage == "custom_hot_start":
                if (step.get("type") != "custom_hot_start"
                        or not step.get("settings", {}).get("custom_hot_start_enabled")):
                    raise ValueError("Generated custom hot start stage is invalid")
            elif stage == "arc_off":
                if (
                    step.get("type") != "digital_weld"
                    or step.get("command") != "off"
                ):
                    raise ValueError("Generated ARC OFF stage cannot be changed")
                if step.get("trigger_before_goal", False):
                    motion_steps = [
                        candidate for candidate in scenario_steps
                        if candidate.get("weld_scenario_stage") == "weld_motion"
                    ]
                    if not motion_steps or int(
                        motion_steps[0].get("parallel_slot", -1)
                    ) != slot:
                        raise ValueError(
                            "Triggered ARC OFF must share the continuous weld-motion slot"
                        )
                elif any(candidate.get("weld_scenario_stage") == "software_crater"
                         for candidate in scenario_steps):
                    if slot <= int(next(candidate for candidate in scenario_steps
                                       if candidate.get("weld_scenario_stage") == "software_crater").get("parallel_slot", 0)):
                        raise ValueError("ARC OFF must follow software crater")
            elif stage == "software_crater":
                if step.get("type") != "software_crater" or not step.get("settings", {}).get("software_crater_enabled"):
                    raise ValueError("Generated software crater stage is invalid")
            elif stage == "start_safe" and step.get("touch_guard", False):
                raise ValueError(
                    "Safe approach motion must not use the Fastech DI0 guard"
                )
            elif stage == "start_contact":
                if step.get("touch_guard", False) and (
                    not step.get("continue_after_touch", False)
                    or step.get("accept_initial_touch", False)
                ):
                    raise ValueError(
                        "Guarded START contact must require a new Fastech DI0 edge, "
                        "stop, then continue"
                    )
            elif stage == "touch_output_off" and (
                step.get("type") != "digital_output"
                or step.get("io_backend") != "fastech_ethernet"
                or int(step.get("port", -1)) != 0
                or bool(step.get("value", True))
            ):
                raise ValueError(
                    "Generated scenario must turn Fastech DO0 OFF before ARC ON"
                )
            elif stage == "touch_output_on" and (
                step.get("type") != "digital_output"
                or step.get("io_backend") != "fastech_ethernet"
                or int(step.get("port", -1)) != 0
                or not bool(step.get("value", False))
            ):
                raise ValueError(
                    "Generated scenario must restore Fastech DO0 ON after finish"
                )
            elif stage == "weld_motion":
                if step.get("touch_guard", False):
                    raise ValueError(
                        "Fastech DI0 must be ignored during weld motion"
                    )
                arc_steps = [
                    candidate for candidate in scenario_steps
                    if candidate.get("weld_scenario_stage") == "arc_on"
                ]
                custom_enabled = bool(arc_steps and arc_steps[0].get(
                    "settings", {}
                ).get("custom_hot_start_enabled", False))
                custom_steps = [
                    candidate for candidate in scenario_steps
                    if candidate.get("weld_scenario_stage") == "custom_hot_start"
                ]
                if custom_enabled != bool(custom_steps):
                    raise ValueError("Custom hot start stage does not match ARC ON settings")
                if custom_steps and not (
                    int(arc_steps[0].get("parallel_slot", -1))
                    < int(custom_steps[0].get("parallel_slot", -1)) < slot
                ):
                    raise ValueError("Custom hot start must precede weld motion")
                # if not arc_steps or int(
                #     arc_steps[0].get("parallel_slot", -1)
                # ) != slot:
                #     raise ValueError(
                #         "Generated weld motion must share the ARC ON slot"
                #     )
        # if require_complete and seen != set(WELD_SCENARIO_STAGE_ORDER):
        #     missing = [
        #         stage for stage in WELD_SCENARIO_STAGE_ORDER if stage not in seen
        #     ]
        #     raise ValueError(
        #         "Generated weld scenario is incomplete: " + ", ".join(missing)
        #     )
    return True
