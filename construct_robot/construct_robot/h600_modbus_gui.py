import csv
import signal
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from construct_msgs.msg import ModbusTrace, WelderStatus
from construct_msgs.srv import SetWelderCommand


class H600GuiNode(Node):
    """ROS adapter for the H600 diagnostic console."""

    def __init__(self, ui):
        super().__init__("h600_modbus_gui")
        self.ui = ui
        self.command_client = self.create_client(
            SetWelderCommand,
            "/h600/set_command",
        )
        self.create_subscription(
            WelderStatus,
            "/h600/status",
            self._status,
            20,
        )
        self.create_subscription(
            ModbusTrace,
            "/h600/traffic",
            self._trace,
            100,
        )

    def _status(self, message):
        self.ui.post(self.ui.update_status, message)

    def _trace(self, message):
        self.ui.post(self.ui.add_trace, message)

    def send_command(self, values):
        if not self.command_client.wait_for_service(timeout_sec=2.0):
            self.ui.post(
                self.ui.set_result,
                False,
                "/h600/set_command unavailable",
            )
            return
        request = SetWelderCommand.Request()
        (
            request.robot_ready,
            request.gas,
            request.arc,
            request.allow_nonzero_setpoints,
            request.current_raw,
            request.voltage_raw,
            request.v_offset_raw,
        ) = values
        future = self.command_client.call_async(request)
        future.add_done_callback(self._command_result)

    def _command_result(self, future):
        try:
            response = future.result()
            self.ui.post(
                self.ui.set_result,
                response.success,
                response.message,
            )
        except Exception as error:
            self.ui.post(self.ui.set_result, False, str(error))


class H600ModbusGui:
    """Tk-based H600 command, feedback, and Modbus packet inspector."""

    TRACE_COLUMNS = (
        "time",
        "direction",
        "client",
        "tx",
        "unit",
        "function",
        "register",
        "count",
        "summary",
    )

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("H600 Modbus TCP Diagnostic Console")
        self.root.geometry("1260x820")
        self._closing = False
        self.trace_rows = []
        self.trace_raw = {}
        self.paused = tk.BooleanVar(value=False)
        self.auto_scroll = tk.BooleanVar(value=True)
        self.robot_ready = tk.BooleanVar(value=False)
        self.gas = tk.BooleanVar(value=False)
        self.arc = tk.BooleanVar(value=False)
        self.nonzero_unlock = tk.BooleanVar(value=False)
        self.current_raw = tk.IntVar(value=0)
        self.voltage_raw = tk.IntVar(value=0)
        self.v_offset_raw = tk.IntVar(value=0)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Sans", 17, "bold"))
        style.configure("Section.TLabel", font=("Sans", 11, "bold"))

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            outer,
            text="H600 Modbus TCP Diagnostic Console",
            style="Title.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            outer,
            text=(
                "ROS command service ↔ register image ↔ H600 Modbus client · "
                "ARC output remains bridge safety-gated"
            ),
        ).pack(anchor=tk.W, pady=(2, 10))

        connection = ttk.LabelFrame(outer, text="Connection / feedback")
        connection.pack(fill=tk.X)
        self.connection_label = ttk.Label(
            connection,
            text="DISCONNECTED · waiting for /h600/status",
        )
        self.connection_label.grid(
            row=0,
            column=0,
            columnspan=5,
            sticky=tk.W,
            padx=8,
            pady=6,
        )
        self.status_fields = {}
        labels = (
            ("201 Ready", "ready"),
            ("202 Gas", "gas"),
            ("202 ARC", "arc"),
            ("211 Status raw", "status"),
            ("211 Error bit7", "error"),
            ("211 Welding bit5", "welding"),
            ("211 Touch bit4", "touch"),
            ("212 Current FB", "current_fb"),
            ("213 Voltage FB", "voltage_fb"),
        )
        for index, (label, key) in enumerate(labels):
            row = 1 + index // 5
            column = index % 5
            frame = ttk.Frame(connection)
            frame.grid(
                row=row,
                column=column,
                sticky=tk.EW,
                padx=8,
                pady=5,
            )
            ttk.Label(frame, text=label).pack(anchor=tk.W)
            value = ttk.Label(frame, text="–", font=("Monospace", 11, "bold"))
            value.pack(anchor=tk.W)
            self.status_fields[key] = value
            connection.columnconfigure(column, weight=1)

        command = ttk.LabelFrame(outer, text="Command register image")
        command.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(
            command,
            text="201 robot ready",
            variable=self.robot_ready,
        ).grid(row=0, column=0, padx=8, pady=8, sticky=tk.W)
        ttk.Checkbutton(
            command,
            text="202 bit3 gas",
            variable=self.gas,
        ).grid(row=0, column=1, padx=8, pady=8, sticky=tk.W)
        ttk.Checkbutton(
            command,
            text="202 bit0 ARC",
            variable=self.arc,
        ).grid(row=0, column=2, padx=8, pady=8, sticky=tk.W)
        ttk.Checkbutton(
            command,
            text="I understand nonzero setpoints",
            variable=self.nonzero_unlock,
        ).grid(row=0, column=3, padx=8, pady=8, sticky=tk.W)

        for column, (label, variable, address) in enumerate(
            (
                ("Current raw", self.current_raw, 204),
                ("Voltage raw", self.voltage_raw, 205),
                ("V-offset raw", self.v_offset_raw, 206),
            )
        ):
            frame = ttk.Frame(command)
            frame.grid(row=1, column=column, padx=8, pady=4, sticky=tk.W)
            ttk.Label(frame, text=f"{label} · register {address}").pack(
                anchor=tk.W
            )
            ttk.Spinbox(
                frame,
                from_=0,
                to=65535,
                textvariable=variable,
                width=12,
            ).pack(anchor=tk.W)

        buttons = ttk.Frame(command)
        buttons.grid(row=1, column=3, padx=8, pady=4, sticky=tk.W)
        ttk.Button(
            buttons,
            text="Send register image",
            command=self.send,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            buttons,
            text="FORCE ARC OFF",
            command=self.force_off,
        ).pack(side=tk.LEFT)
        self.command_result = ttk.Label(
            command,
            text="No command sent",
        )
        self.command_result.grid(
            row=2,
            column=0,
            columnspan=4,
            padx=8,
            pady=(3, 8),
            sticky=tk.W,
        )

        trace_header = ttk.Frame(outer)
        trace_header.pack(fill=tk.X, pady=(12, 5))
        ttk.Label(
            trace_header,
            text="Modbus TCP traffic",
            style="Section.TLabel",
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            trace_header,
            text="Pause display",
            variable=self.paused,
        ).pack(side=tk.LEFT, padx=(18, 4))
        ttk.Checkbutton(
            trace_header,
            text="Auto scroll",
            variable=self.auto_scroll,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            trace_header,
            text="Clear",
            command=self.clear_trace,
        ).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(
            trace_header,
            text="Export CSV",
            command=self.export_csv,
        ).pack(side=tk.RIGHT)

        trace_frame = ttk.Frame(outer)
        trace_frame.pack(fill=tk.BOTH, expand=True)
        self.trace = ttk.Treeview(
            trace_frame,
            columns=self.TRACE_COLUMNS,
            show="headings",
            height=13,
        )
        widths = (95, 55, 145, 55, 45, 65, 75, 55, 300)
        for name, width in zip(self.TRACE_COLUMNS, widths):
            self.trace.heading(name, text=name.upper())
            self.trace.column(name, width=width, anchor=tk.W)
        scrollbar = ttk.Scrollbar(
            trace_frame,
            orient=tk.VERTICAL,
            command=self.trace.yview,
        )
        self.trace.configure(yscrollcommand=scrollbar.set)
        self.trace.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.trace.tag_configure("RX", foreground="#0066aa")
        self.trace.tag_configure("TX", foreground="#8a2be2")
        self.trace.bind("<<TreeviewSelect>>", self.show_raw_frame)

        ttk.Label(outer, text="Selected MBAP + PDU raw bytes").pack(
            anchor=tk.W,
            pady=(7, 2),
        )
        self.raw = tk.Text(
            outer,
            height=4,
            wrap=tk.WORD,
            bg="#101820",
            fg="#d5f5e3",
            font=("Monospace", 10),
        )
        self.raw.pack(fill=tk.X)

        self.node = H600GuiNode(self)
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(
            target=self.executor.spin,
            daemon=True,
        )
        self.executor_thread.start()
        signal.signal(
            signal.SIGINT,
            lambda _signum, _frame: self.root.after(0, self.close),
        )
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(200, self.check_ros)

    def post(self, callback, *args):
        self.root.after(0, callback, *args)

    @staticmethod
    def _bool_text(value):
        return "ON" if value else "OFF"

    def update_status(self, message):
        state = "CONNECTED" if message.client_connected else "DISCONNECTED"
        address = message.client_address or "no Modbus client"
        self.connection_label.configure(text=f"{state} · {address}")
        values = {
            "ready": self._bool_text(message.robot_ready),
            "gas": self._bool_text(message.gas),
            "arc": self._bool_text(message.arc),
            "status": (
                f"{message.status_raw} / 0x{message.status_raw:04X}"
            ),
            "error": self._bool_text(message.welder_error),
            "welding": self._bool_text(message.welding),
            "touch": self._bool_text(message.touch_detect),
            "current_fb": str(message.current_feedback_raw),
            "voltage_fb": str(message.voltage_feedback_raw),
        }
        for key, value in values.items():
            self.status_fields[key].configure(text=value)

    def command_values(self):
        try:
            raw_values = (
                int(self.current_raw.get()),
                int(self.voltage_raw.get()),
                int(self.v_offset_raw.get()),
            )
        except (ValueError, tk.TclError):
            raise ValueError("Raw setpoints must be uint16 integers")
        if any(value < 0 or value > 65535 for value in raw_values):
            raise ValueError("Raw setpoints must be in 0..65535")
        return (
            bool(self.robot_ready.get()),
            bool(self.gas.get()),
            bool(self.arc.get()),
            bool(self.nonzero_unlock.get()),
        ) + raw_values

    def send(self):
        try:
            values = self.command_values()
        except ValueError as error:
            self.set_result(False, str(error))
            return
        self.command_result.configure(text="Sending…")
        threading.Thread(
            target=self.node.send_command,
            args=(values,),
            daemon=True,
        ).start()

    def force_off(self):
        self.robot_ready.set(False)
        self.gas.set(False)
        self.arc.set(False)
        self.current_raw.set(0)
        self.voltage_raw.set(0)
        self.v_offset_raw.set(0)
        self.nonzero_unlock.set(False)
        self.send()

    def set_result(self, success, message):
        prefix = "OK" if success else "REJECTED"
        color = "#137333" if success else "#b3261e"
        self.command_result.configure(
            text=f"{prefix} · {message}",
            foreground=color,
        )

    def add_trace(self, message):
        timestamp = time.strftime(
            "%H:%M:%S",
            time.localtime(message.header.stamp.sec),
        )
        milliseconds = message.header.stamp.nanosec // 1_000_000
        row = {
            "time": f"{timestamp}.{milliseconds:03d}",
            "direction": message.direction,
            "client": message.client_address,
            "tx": message.transaction_id,
            "unit": message.unit_id,
            "function": f"0x{message.function_code:02X}",
            "register": message.register_address,
            "count": message.register_count,
            "summary": message.summary,
            "raw_hex": message.raw_hex,
        }
        self.trace_rows.append(row)
        if len(self.trace_rows) > 10000:
            self.trace_rows.pop(0)
        if self.paused.get():
            return
        item = self.trace.insert(
            "",
            tk.END,
            values=tuple(row[name] for name in self.TRACE_COLUMNS),
            tags=(message.direction,),
        )
        self.trace_raw[item] = message.raw_hex
        if len(self.trace.get_children()) > 2000:
            oldest = self.trace.get_children()[0]
            self.trace_raw.pop(oldest, None)
            self.trace.delete(oldest)
        if self.auto_scroll.get():
            self.trace.see(item)

    def show_raw_frame(self, _event=None):
        selection = self.trace.selection()
        if not selection:
            return
        value = self.trace_raw.get(selection[0], "")
        self.raw.delete("1.0", tk.END)
        self.raw.insert(tk.END, value)

    def clear_trace(self):
        self.trace_rows.clear()
        self.trace_raw.clear()
        self.trace.delete(*self.trace.get_children())
        self.raw.delete("1.0", tk.END)

    def export_csv(self):
        path = filedialog.asksaveasfilename(
            title="Export Modbus trace",
            defaultextension=".csv",
            filetypes=(("CSV", "*.csv"), ("All files", "*.*")),
        )
        if not path:
            return
        fields = self.TRACE_COLUMNS + ("raw_hex",)
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.trace_rows)
        self.set_result(True, f"Exported {len(self.trace_rows)} rows to {path}")

    def close(self):
        if self._closing:
            return
        self._closing = True
        self.root.quit()
        self.root.destroy()

    def shutdown_ros(self):
        if rclpy.ok():
            rclpy.shutdown()
        self.executor_thread.join(timeout=1.0)
        self.node.destroy_node()

    def check_ros(self):
        if not rclpy.ok():
            self.close()
            return
        self.root.after(200, self.check_ros)

    def mainloop(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    gui = H600ModbusGui()
    try:
        gui.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        gui.shutdown_ros()
