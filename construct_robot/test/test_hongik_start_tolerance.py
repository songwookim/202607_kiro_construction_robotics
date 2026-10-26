"""Field start tolerance follows the parameter server and caches its source; no motion."""

from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def adapter(monkeypatch):
    stack = Path(__file__).resolve().parents[2] / "hongik_fine_sensing_field"
    for name in ("scan_core", "scan_l0", "vc3dz_driver", "hongik_scan_ros"):
        monkeypatch.syspath_prepend(str(stack / name))
    from hongik_scan_ros import moveit_adapter
    return moveit_adapter


def platform(adapter):
    p = adapter.MoveItPlatform.__new__(adapter.MoveItPlatform)
    clients = []
    p.mj = NS(plan_service="/cell/plan_kinematic_path")
    p.cb = object()
    def create_client(service, name, **kwargs):
        clients.append((name, kwargs))
        return object()
    p.ctx = NS(node=NS(create_client=create_client))
    return p, clients


def test_configured_tolerance_uses_planner_namespace_and_is_read_once(adapter, monkeypatch):
    p, clients = platform(adapter)
    requests = []
    def read(client, request, *args, **kwargs):
        requests.append(request.names)
        return NS(values=[NS(type=3, double_value=0.002)])
    monkeypatch.setattr(adapter, "call_service", read)
    assert p.allowed_start_tolerance() == (0.002, "move_group trajectory_execution.allowed_start_tolerance")
    assert p.allowed_start_tolerance() == (0.002, "move_group trajectory_execution.allowed_start_tolerance")
    assert requests == [["trajectory_execution.allowed_start_tolerance"]]
    assert clients == [("/cell/move_group/get_parameters", {"callback_group": p.cb})]


@pytest.mark.parametrize("response", [None, NS(values=[]), NS(values=[NS(type=0, double_value=0.0)]),
                                      NS(values=[NS(type=3, double_value=0.0)])])
def test_missing_or_disabled_tolerance_uses_reported_default(adapter, monkeypatch, response):
    p, _clients = platform(adapter)
    monkeypatch.setattr(adapter, "call_service", lambda *args, **kwargs: response)
    tolerance, source = p.allowed_start_tolerance()
    assert tolerance == 0.01
    assert source.startswith("MoveIt default allowed_start_tolerance 0.01 rad (")


def test_failed_parameter_lookup_keeps_fallback_reason(adapter, monkeypatch):
    p, _clients = platform(adapter)
    def fail(*args, **kwargs):
        raise RuntimeError("service unavailable")
    monkeypatch.setattr(adapter, "call_service", fail)
    tolerance, source = p.allowed_start_tolerance()
    assert tolerance == 0.01
    assert "RuntimeError: service unavailable" in source
