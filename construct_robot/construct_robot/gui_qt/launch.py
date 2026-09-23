"""Standalone Qt preview entry point; imports no ROS or Tkinter runtime."""


def main():
    try:
        from .main_window import launch
    except ImportError as error:
        if error.name and error.name.startswith("PySide6"):
            raise SystemExit("PySide6 is required: install construct_robot[qt]") from error
        raise
    return launch()


if __name__ == "__main__":
    main()
