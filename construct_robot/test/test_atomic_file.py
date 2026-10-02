"""File persistence tests use temporary folders; no teaching files are touched."""

from pathlib import Path

import pytest
import yaml

from construct_robot.io.teaching_yaml import atomic_text_writer, atomic_yaml


def test_atomic_yaml_preserves_format_and_replaces_existing_file(tmp_path):
    path = tmp_path / "teaching" / "pass_1.yaml"
    document = {"z": "교시", "a": [1.0, 2.0]}
    for unicode in (False, True):
        atomic_yaml(path, document, allow_unicode=unicode)
        assert path.read_text(encoding="utf-8") == yaml.safe_dump(
            document, sort_keys=False, allow_unicode=unicode,
        )
        assert yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader) == document
        assert list(path.parent.iterdir()) == [path]


def test_writer_failure_preserves_old_target_and_cleans_temporary(tmp_path):
    path = tmp_path / "target.log"
    path.write_text("previous")
    with pytest.raises(ValueError, match="format error"):
        with atomic_text_writer(path) as stream:
            stream.write("partial report")
            assert path.read_text() == "previous"
            raise ValueError("format error")
    assert path.read_text() == "previous"
    assert list(tmp_path.iterdir()) == [path]


def test_replace_failure_preserves_old_target_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / "target.yaml"
    path.write_text("previous")

    def fail_replace(_temporary, _target):
        raise OSError("replacement failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        atomic_yaml(path, {"new": True})
    assert path.read_text() == "previous"
    assert list(tmp_path.iterdir()) == [path]


def test_yaml_serialization_failure_does_not_leave_a_partial_new_file(tmp_path):
    path = tmp_path / "new.yaml"
    with pytest.raises(yaml.representer.RepresenterError):
        atomic_yaml(path, {"unsupported": object()})
    assert not path.exists()
    assert not list(tmp_path.iterdir())
