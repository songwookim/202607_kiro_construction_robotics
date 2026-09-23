"""Source teaching YAML directory shared by the GUI panels."""

from pathlib import Path


def teaching_config_dir():
    source = Path(__file__).resolve().parents[2] / "construct_description" / "config"
    if source.is_dir():
        return source
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("construct_description")) / "config"
