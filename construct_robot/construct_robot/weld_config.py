"""GUI-independent, protocol-validated welding configuration."""
import copy
import math

from .hicomm_welder import TxState, build_request

DIGITAL_WELD_RECIPE_KEYS = (
    "current_a",
    "voltage_tenths",
    "material",
    "diameter_mm",
    "mode",
    "gas",
    "synergic",
    "correction",
    "hot_start_current_a",
    "hot_start_hold_adjustment",
)
DIGITAL_WELD_COMMANDS = frozenset(("set", "on", "off"))

DEFAULT_DIGITAL_WELD_SETTINGS = {
    # Current production/test recipe. Build Scenario snapshots the live GUI
    # values, so changing the GUI before Build still overrides these defaults.
    "current_a": 200,
    "voltage_tenths": 250,
    "voltage": 25.0,
    "material": "FE-SOLID",
    "diameter_mm": 1.2,
    "mode": "LSM",
    "gas": "CO2",
    "synergic": False,
    "correction": 0.0,
    # Hot Start is encoded in Hi-COMM TX; crater values below are panel-only
    # references and must not become a PC current profile.
    "hot_start_enabled": True,
    "hot_start_percent": 20.0,
    "hot_start_hold_adjustment": 0,
    "custom_hot_start_enabled": True,
    "custom_hot_start_hold_s": 0.15,
    "custom_hot_start_percent": 20.0,
    "expect_native_crater": True,
    # The shared Hi-COMM TX table has no crater setpoint fields. These two
    # values mirror the welder-panel test recipe and are logged as references.
    "crater_panel_current_ref_a": 60.0,
    "crater_panel_voltage_ref_v": 25.0,
    "crater_panel_time_ref_s": 1.0,
    "software_crater_enabled": False,
    "software_crater_ratio_percent": 30.0,
    "software_crater_voltage_v": 25.0,
    "software_crater_hold_s": 0.5,
    # Added to the integrated wire-feed estimate. Keep at zero until a
    # measured torch/liner run-out allowance has been calibrated.
    "wire_consumable_alpha_mm": 0.0,
}

def digital_weld_recipe(settings):
    """Return only the values encoded into the Hi-COMM welding frame."""
    recipe = {key: settings[key] for key in DIGITAL_WELD_RECIPE_KEYS}
    # Gas timing controls were deliberately removed from the application;
    # transmit zero explicitly so a prior recipe cannot leak into a test.
    recipe.update(pre_gas_s=0.0, post_gas_s=0.0)
    return recipe


def validate_digital_weld_settings(settings):
    """Normalize and validate GUI/sequence digital-welding settings."""
    normalized = copy.deepcopy(DEFAULT_DIGITAL_WELD_SETTINGS)
    # Accept older saved scenarios while exposing only reference-only crater
    # names to new callers. These values never enter digital_weld_recipe().
    legacy_names = {
        "crater_enabled": "expect_native_crater",
        "crater_current_a": "crater_panel_current_ref_a",
        "crater_voltage_v": "crater_panel_voltage_ref_v",
        "crater_seconds": "crater_panel_time_ref_s",
    }
    normalized.update({legacy_names.get(key, key): value
                       for key, value in settings.items()})
    for removed_key in ("pre_gas_s", "post_gas_s", "preflow_seconds"):
        normalized.pop(removed_key, None)
    normalized["current_a"] = int(round(float(normalized["current_a"])))
    normalized["voltage_tenths"] = int(round(
        float(normalized["voltage_tenths"])
    ))
    normalized["voltage"] = normalized["voltage_tenths"] / 10.0
    normalized["diameter_mm"] = float(normalized["diameter_mm"])
    normalized["synergic"] = bool(normalized["synergic"])
    for key in (
        "correction",
        "hot_start_percent", "hot_start_hold_adjustment",
        "custom_hot_start_hold_s", "custom_hot_start_percent",
        "crater_panel_time_ref_s",
        "crater_panel_current_ref_a", "crater_panel_voltage_ref_v",
        "software_crater_ratio_percent", "software_crater_voltage_v",
        "software_crater_hold_s",
        "wire_consumable_alpha_mm",
    ):
        normalized[key] = float(normalized[key])
    normalized["hot_start_enabled"] = bool(normalized["hot_start_enabled"])
    normalized["custom_hot_start_enabled"] = bool(normalized["custom_hot_start_enabled"])
    boost = normalized["custom_hot_start_percent"]
    if not math.isfinite(boost) or not 0 <= boost <= 100:
        raise ValueError("Custom hot start boost must be in 0..100 percent")
    if normalized["custom_hot_start_enabled"] and not 30 <= round(normalized["current_a"] * (1 + boost / 100)) <= 400:
        raise ValueError("Custom hot start boosted current must be in 30..400 A")
    if not 0.01 <= normalized["custom_hot_start_hold_s"] <= 5.0:
        raise ValueError("custom hot start hold must be in 0.01..5.0 seconds")
    normalized["expect_native_crater"] = bool(normalized["expect_native_crater"])
    normalized["software_crater_enabled"] = bool(normalized["software_crater_enabled"])
    if not 20.0 <= normalized["software_crater_ratio_percent"] <= 40.0:
        raise ValueError("software crater ratio must be in 20..40 percent")
    if not 10.0 <= normalized["software_crater_voltage_v"] <= 40.0:
        raise ValueError("software crater voltage must be in 10.0..40.0 V")
    if not 0.0 < normalized["software_crater_hold_s"] <= 5.0:
        raise ValueError("software crater hold must be in (0, 5] seconds")
    crater_current = round(normalized["current_a"] * normalized["software_crater_ratio_percent"] / 100.0)
    if normalized["software_crater_enabled"] and not 30 <= crater_current <= 400:
        raise ValueError("software crater current must be in 30..400 A")
    if not 0.0 <= normalized["hot_start_percent"] <= 100.0:
        raise ValueError("hot-start boost must be in 0..100 percent")
    normalized["hot_start_hold_adjustment"] = int(round(
        normalized["hot_start_hold_adjustment"]
    ))
    if not -15 <= normalized["hot_start_hold_adjustment"] <= 15:
        raise ValueError("hot-start hold adjustment must be in -15..15")
    normalized.pop("crater_percent", None)
    if not 0.0 <= normalized["crater_panel_current_ref_a"] <= 600.0:
        raise ValueError("crater panel current must be in 0..600 A")
    if not 3.0 <= normalized["crater_panel_voltage_ref_v"] <= 80.0:
        raise ValueError("crater panel voltage must be in 3.0..80.0 V")
    if not 0.0 <= normalized["crater_panel_time_ref_s"] <= 30.0:
        raise ValueError("crater panel time reference must be in 0..30 seconds")
    if not -1000.0 <= normalized["wire_consumable_alpha_mm"] <= 1000.0:
        raise ValueError("wire consumable alpha must be in -1000..1000 mm")
    profile = weld_current_profile(normalized)
    normalized["hot_start_current_a"] = (
        profile["hot"] if normalized["hot_start_enabled"] else 0
    )
    # build_request is the protocol's single source of range/enum validation.
    build_request(TxState(**digital_weld_recipe(normalized)))
    return normalized


def weld_current_profile(settings):
    """Return only PC-commanded nominal and native Hot Start currents."""
    nominal = int(round(float(settings["current_a"])))
    hot = nominal
    if bool(settings.get("hot_start_enabled", True)):
        hot = int(round(
            nominal * (1.0 + float(settings.get("hot_start_percent", 20.0)) / 100.0)
        ))
    return {"nominal": nominal, "hot": hot}


DEFAULT_WELD_MOTION_SETTINGS = {
    "weld_tcp_speed_mm_s": 3.0,
    "weld_weave_enabled": False,
    "weld_weave_pattern": "sine",
    "weld_weave_amplitude_mm": 3.0,
    "weld_weave_pitch_mm": 5.0,
    "weld_weave_left_dwell_s": 0.0,
    "weld_weave_right_dwell_s": 0.0,
    "weld_weave_axis": "tool_y",
}


class WeldConfigurationState:
    """Offline recipe/motion draft; editing it never transmits Hi-COMM data."""

    def __init__(self, recipe=None, motion=None):
        self.recipe = validate_digital_weld_settings(recipe or {})
        self.motion = dict(DEFAULT_WELD_MOTION_SETTINGS)
        if motion:
            for key, value in motion.items():
                if key in self.motion:
                    self.set_motion(key, value)

    def replace(self, recipe, motion):
        """Atomically replace an offline draft from current builder defaults."""
        candidate = WeldConfigurationState(recipe, motion)
        self.recipe = candidate.recipe
        self.motion = candidate.motion

    def set_recipe(self, key, value):
        if key not in DEFAULT_DIGITAL_WELD_SETTINGS or key in ("voltage",):
            raise ValueError(f"Unsupported weld setting: {key}")
        candidate = dict(self.recipe)
        candidate[key] = value
        self.recipe = validate_digital_weld_settings(candidate)

    def set_motion(self, key, value):
        if key not in self.motion:
            raise ValueError(f"Unsupported weld motion setting: {key}")
        if key == "weld_weave_enabled":
            normalized = bool(value)
        elif key == "weld_weave_pattern":
            if value not in ("sine", "crescent", "circle"):
                raise ValueError("Unsupported weave pattern")
            normalized = value
        elif key == "weld_weave_axis":
            if value not in ("tool_x", "tool_y"):
                raise ValueError("Unsupported weave axis")
            normalized = value
        else:
            normalized = float(value)
            bounds = {
                "weld_tcp_speed_mm_s": (0.1, 100.0),
                "weld_weave_amplitude_mm": (0.1, 50.0),
                "weld_weave_pitch_mm": (0.1, 100.0),
                "weld_weave_left_dwell_s": (0.0, 10.0),
                "weld_weave_right_dwell_s": (0.0, 10.0),
            }
            minimum, maximum = bounds[key]
            if not math.isfinite(normalized) or not minimum <= normalized <= maximum:
                raise ValueError(f"{key} must be in {minimum:g}..{maximum:g}")
        self.motion[key] = normalized
