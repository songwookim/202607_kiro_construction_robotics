from types import SimpleNamespace
from unittest.mock import Mock

from construct_robot.gui import weld_parameter_panel
from test_weld_parameter_table import LOG


def panel_for(path):
    panel = object.__new__(weld_parameter_panel.WeldParameterPanel)
    panel.host = SimpleNamespace(weld_current_raw=SimpleNamespace(get=lambda: 245))
    panel.latest_log_path = path
    panel._refresh_pending = None
    panel._log_signature = None
    panel._last_log = {}
    panel._rendered_rows = []
    panel._row_items = []
    panel._source_text = None
    panel.source_label = Mock()
    panel.tree = Mock()
    panel.tree.get_children.return_value = ()
    panel.tree.insert.side_effect = (f"item_{index}" for index in range(500))
    return panel


def test_setting_edits_reuse_log_and_update_only_changed_cells(tmp_path, monkeypatch):
    path = tmp_path / "latest.log"
    path.write_text(LOG, encoding="utf-8")
    read = Mock(wraps=weld_parameter_panel.read_weld_log_sections)
    monkeypatch.setattr(weld_parameter_panel, "read_weld_log_sections", read)
    panel = panel_for(path)
    panel.refresh()
    panel.tree.reset_mock()
    panel.refresh()
    panel.tree.item.assert_not_called()
    panel.host.weld_current_raw.get = lambda: 250
    panel.refresh()
    read.assert_called_once_with(path)
    panel.tree.delete.assert_not_called()
    panel.tree.insert.assert_not_called()
    panel.tree.item.assert_called_once()
    update = panel.tree.item.call_args.kwargs
    assert update["values"][1:3] == ("250", "245")
    assert update["tags"] == ("changed",)


def test_replaced_and_deleted_logs_invalidate_cached_results(tmp_path):
    path = tmp_path / "latest.log"
    path.write_text(LOG, encoding="utf-8")
    panel = panel_for(path)
    panel.refresh()
    replacement = tmp_path / "replacement.log"
    replacement.write_text(LOG.replace("current_a=245", "current_a=246"), encoding="utf-8")
    replacement.replace(path)
    panel.refresh()
    assert panel._last_log["commanded"]["current_a"] == "246"
    path.unlink()
    panel.refresh()
    assert panel._last_log == {}
    assert panel._source_text == "직전 용접 로그 없음"
    path.write_text(LOG, encoding="utf-8")
    panel.refresh()
    assert panel._last_log["commanded"]["current_a"] == "245"


def test_task_switch_reads_the_selected_tasks_log(tmp_path):
    first = tmp_path / "1g.log"
    second = tmp_path / "3g.log"
    first.write_text(LOG, encoding="utf-8")
    second.write_text(LOG.replace("current_a=245", "current_a=180"), encoding="utf-8")
    panel = panel_for(first)
    panel.refresh()
    panel.latest_log_path = second
    panel.refresh()
    assert panel._last_log["commanded"]["current_a"] == "180"


def test_manual_refresh_cancels_the_pending_automatic_refresh(tmp_path):
    panel = panel_for(tmp_path / "missing.log")
    panel._refresh_pending = "pending"
    panel.refresh()
    panel.tree.after_cancel.assert_called_once_with("pending")
    assert panel._refresh_pending is None
