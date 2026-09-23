"""Qt production window with the existing, visible Tk runtime as sole owner.

Both event loops are pumped on the main thread.  No second ROS node, TCP
client, controller, or I/O owner is constructed for Qt.
"""


def main(args=None):
    import tkinter as tk

    import rclpy
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from construct_robot.production_runtime import ProductionRuntimePort
    from construct_robot.weld_action_gui import WeldActionGui
    from .main_window import SequenceMainWindow

    app = QApplication.instance() or QApplication([])
    rclpy.init(args=args)
    gui = None
    timer = None
    try:
        gui = WeldActionGui()
        gui.root.title("Welding production runtime · Tk safety host")
        port = ProductionRuntimePort(gui)
        window = SequenceMainWindow(
            port.sequence_state, port,
            teaching_state=port.teaching_state,
            multipass_state=port.multipass_state,
        )
        timer = QTimer(window)
        timer.setInterval(10)

        def pump_tk():
            if getattr(gui, "_closing", False):
                app.quit()
                return
            try:
                gui.root.update()
                if not gui.root.winfo_exists():
                    app.quit()
            except tk.TclError:
                # The existing Tk close path owns shutdown; stop Qt as well.
                app.quit()

        timer.timeout.connect(pump_tk)
        timer.start()
        window.show()
        return app.exec()
    finally:
        if timer is not None:
            timer.stop()
        if gui is not None:
            gui.close()
            gui.shutdown_ros()
        elif rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
