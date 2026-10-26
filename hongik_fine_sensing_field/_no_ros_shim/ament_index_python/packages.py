import os

# .../src/scan_stack/_no_ros_shim/ament_index_python/packages.py → .../src
SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class PackageNotFoundError(KeyError):
    pass


def get_package_share_directory(name):
    """src/ 아래에서 <name>name</name> 을 가진 package.xml 의 디렉터리를 반환 (share 디렉터리 대용)."""
    for root, dirs, files in os.walk(SRC_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("build", "install", "log")]
        if "package.xml" in files:
            with open(os.path.join(root, "package.xml"), "r", encoding="utf-8") as f:
                if "<name>%s</name>" % name in f.read():
                    return root
    raise PackageNotFoundError("package '%s' not found under %s" % (name, SRC_DIR))
