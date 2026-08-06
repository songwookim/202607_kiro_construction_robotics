import csv
import signal
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from construct_msgs.msg import ModbusTrace, WelderStatus
from construct_msgs.srv import (
    GetModbusRegisters,
    SetModbusServer,
    SetWelderCommand,
)


H600_PORT = 502
TRACE_QOS = QoSProfile(
    depth=200,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class H600GuiNode(Node):
    """ROS adapter for the H600 diagnostic console."""

    def __init__(self, ui):
        super().__init__("h600_modbus_gui")
        self.ui = ui
        self.command_client = self.create_client(
            SetWelderCommand,
            "/h600/set_command",
        )
        self.server_client = self.create_client(
            SetModbusServer,
            "/h600/set_server",
        )
        self.register_client = self.create_client(
            GetModbusRegisters,
            "/h600/get_registers",
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
            TRACE_QOS,
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
            request.robot_error,
            request.touch,
            request.gas,
            request.reverse_inching,
            request.inching,
            request.arc,
            request.allow_nonzero_setpoints,
            request.current_raw,
            request.voltage_raw,
            request.v_offset_raw,
        ) = values
        future = self.command_client.call_async(request)
        future.add_done_callback(self._command_result)

    def set_server(self, start, host, port):
        if not self.server_client.wait_for_service(timeout_sec=2.0):
            self.ui.post(self.ui.set_server_result, False, "/h600/set_server unavailable")
            return
        request = SetModbusServer.Request()
        request.start = start
        request.host = host
        request.port = port
        future = self.server_client.call_async(request)
        future.add_done_callback(self._server_result)

    def _server_result(self, future):
        try:
            response = future.result()
            self.ui.post(self.ui.set_server_result, response.success, response.message)
        except Exception as error:
            self.ui.post(self.ui.set_server_result, False, str(error))

    def get_registers(self, start, count):
        if not self.register_client.wait_for_service(timeout_sec=1.0):
            self.ui.post(self.ui.set_register_result, False, "/h600/get_registers unavailable")
            return
        request = GetModbusRegisters.Request()
        request.start_address = start
        request.count = count
        future = self.register_client.call_async(request)
        future.add_done_callback(
            lambda result: self._register_result(result, start)
        )

    def _register_result(self, future, start):
        try:
            response = future.result()
            self.ui.post(
                self.ui.update_registers,
                response.success,
                response.message,
                start,
                list(response.values),
            )
        except Exception as error:
            self.ui.post(self.ui.set_register_result, False, str(error))

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
        self.root.geometry("1260x900")
        self._closing = False
        self.trace_rows = []
        self.trace_raw = {}
        self.paused = tk.BooleanVar(value=False)
        self.auto_scroll = tk.BooleanVar(value=True)
        self.robot_ready = tk.BooleanVar(value=False)
        self.robot_error = tk.BooleanVar(value=False)
        self.touch = tk.BooleanVar(value=False)
        self.gas = tk.BooleanVar(value=False)
        self.reverse_inching = tk.BooleanVar(value=False)
        self.inching = tk.BooleanVar(value=False)
        self.arc = tk.BooleanVar(value=False)
        self.nonzero_unlock = tk.BooleanVar(value=False)
        self.current_raw = tk.IntVar(value=0)
        self.voltage_raw = tk.IntVar(value=0)
        self.v_offset_raw = tk.IntVar(value=0)
        self.bind_host = tk.StringVar(value="0.0.0.0")
        self.register_start = tk.IntVar(value=201)
        self.register_end = tk.IntVar(value=216)
        self.register_nonzero = tk.BooleanVar(value=False)
        self.register_auto = tk.BooleanVar(value=True)
        self.previous_registers = {}
        self._server_fields_synced = False

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

        server = ttk.LabelFrame(outer, text="Modbus TCP server")
        server.pack(fill=tk.X)
        ttk.Label(server, text="Bind host").grid(row=0, column=0, padx=(8, 3), pady=7)
        ttk.Entry(server, textvariable=self.bind_host, width=16).grid(row=0, column=1, padx=3)
        ttk.Label(server, text="Port").grid(row=0, column=2, padx=(12, 3))
        ttk.Label(
            server,
            text="502 (fixed)",
            font=("Monospace", 10, "bold"),
        ).grid(row=0, column=3, padx=3)
        ttk.Button(
            server,
            text="Start listening",
            command=lambda: self.control_server(True),
        ).grid(row=0, column=4, padx=(12, 3))
        ttk.Button(
            server,
            text="Stop listening",
            command=lambda: self.control_server(False),
        ).grid(row=0, column=5, padx=3)
        self.server_result = ttk.Label(server, text="Waiting for bridge status")
        self.server_result.grid(row=0, column=6, padx=12, sticky=tk.W)

        connection = ttk.LabelFrame(outer, text="H600 connection / feedback")
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
            ("202 Command raw", "control_raw"),
            ("202 Gas", "gas"),
            ("202 ARC", "arc"),
            ("211 Status raw", "status"),
            ("211 Heartbeat", "heartbeat"),
            ("211 Info bits1..0", "info"),
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

        command = ttk.LabelFrame(outer, text="Command registers 201..210")
        command.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(
            command,
            text="201 robot ready",
            variable=self.robot_ready,
            command=self.send,
        ).grid(row=0, column=0, padx=8, pady=8, sticky=tk.W)
        controls = (
            ("202 b7 robot error", self.robot_error),
            ("202 b4 touch", self.touch),
            ("202 b3 gas", self.gas),
            ("202 b2 reverse inch", self.reverse_inching),
            ("202 b1 inch", self.inching),
            ("202 b0 ARC", self.arc),
        )
        for index, (text, variable) in enumerate(controls, start=1):
            name = "reverse" if variable is self.reverse_inching else (
                "forward" if variable is self.inching else "other"
            )
            ttk.Checkbutton(
                command,
                text=text,
                variable=variable,
                command=lambda selected=name: self.control_toggled(selected),
            ).grid(
                row=0, column=index, padx=5, pady=8, sticky=tk.W
            )
        ttk.Checkbutton(
            command,
            text="I understand nonzero setpoints",
            variable=self.nonzero_unlock,
        ).grid(row=1, column=3, columnspan=2, padx=8, pady=8, sticky=tk.W)

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
        buttons.grid(row=1, column=5, columnspan=2, padx=8, pady=4, sticky=tk.W)
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
            columnspan=7,
            padx=8,
            pady=(3, 8),
            sticky=tk.W,
        )

        register_header = ttk.Frame(outer)
        register_header.pack(fill=tk.X, pady=(10, 3))
        ttk.Label(
            register_header,
            text="Holding register monitor",
            style="Section.TLabel",
        ).pack(side=tk.LEFT)
        ttk.Label(register_header, text="Start").pack(side=tk.LEFT, padx=(18, 3))
        ttk.Spinbox(
            register_header, from_=0, to=65535,
            textvariable=self.register_start, width=7,
        ).pack(side=tk.LEFT)
        ttk.Label(register_header, text="End").pack(side=tk.LEFT, padx=(8, 3))
        ttk.Spinbox(
            register_header, from_=0, to=65535,
            textvariable=self.register_end, width=7,
        ).pack(side=tk.LEFT)
        ttk.Button(
            register_header,
            text="Refresh",
            command=self.refresh_registers,
        ).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(
            register_header,
            text="Auto",
            variable=self.register_auto,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            register_header,
            text="Nonzero only",
            variable=self.register_nonzero,
            command=self.refresh_registers,
        ).pack(side=tk.LEFT, padx=6)
        self.register_result = ttk.Label(register_header, text="201..216")
        self.register_result.pack(side=tk.RIGHT)

        register_frame = ttk.Frame(outer)
        register_frame.pack(fill=tk.X)
        self.register_table = ttk.Treeview(
            register_frame,
            columns=("address", "decimal", "hex", "meaning"),
            show="headings",
            height=5,
        )
        for name, width in (("address", 90), ("decimal", 100), ("hex", 100), ("meaning", 650)):
            self.register_table.heading(name, text=name.upper())
            self.register_table.column(name, width=width, anchor=tk.W)
        self.register_table.tag_configure("changed", background="#fff2a8")
        register_scroll = ttk.Scrollbar(
            register_frame,
            orient=tk.VERTICAL,
            command=self.register_table.yview,
        )
        self.register_table.configure(yscrollcommand=register_scroll.set)
        self.register_table.pack(side=tk.LEFT, fill=tk.X, expand=True)
        register_scroll.pack(side=tk.RIGHT, fill=tk.Y)

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
        self.root.after(500, self.auto_refresh_registers)

    def post(self, callback, *args):
        self.root.after(0, callback, *args)

    @staticmethod
    def _bool_text(value):
        return "ON" if value else "OFF"

    def update_status(self, message):
        server_state = "LISTENING" if message.server_running else "STOPPED"
        bind = message.bind_address or "not bound"
        if message.server_running and not self._server_fields_synced:
            try:
                host, port = message.bind_address.rsplit(":", 1)
                self.bind_host.set(host)
                if int(port) != H600_PORT:
                    raise ValueError("Unexpected H600 port")
                self._server_fields_synced = True
            except (ValueError, tk.TclError):
                pass
        self.server_result.configure(
            text=f"{server_state} · {bind}",
            foreground="#137333" if message.server_running else "#b3261e",
        )
        state = "CONNECTED" if message.client_connected else "DISCONNECTED"
        address = message.client_address or "no Modbus client"
        self.connection_label.configure(text=f"{state} · {address}")
        values = {
            "ready": self._bool_text(message.robot_ready),
            "control_raw": (
                f"{self.control_word(message)} / "
                f"0x{self.control_word(message):04X}"
            ),
            "gas": self._bool_text(message.gas),
            "arc": self._bool_text(message.arc),
            "status": (
                f"{message.status_raw} / 0x{message.status_raw:04X}"
            ),
            "heartbeat": str(message.heartbeat),
            "info": str(message.welder_info),
            "error": self._bool_text(message.welder_error),
            "welding": self._bool_text(message.welding),
            "touch": self._bool_text(message.touch_detect),
            "current_fb": str(message.current_feedback_raw),
            "voltage_fb": str(message.voltage_feedback_raw),
        }
        for key, value in values.items():
            self.status_fields[key].configure(text=value)

    @staticmethod
    def control_word(message):
        return (
            (int(message.command_robot_error) << 7)
            | (int(message.command_touch) << 4)
            | (int(message.gas) << 3)
            | (int(message.reverse_inching) << 2)
            | (int(message.inching) << 1)
            | int(message.arc)
        )

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
            bool(self.robot_error.get()),
            bool(self.touch.get()),
            bool(self.gas.get()),
            bool(self.reverse_inching.get()),
            bool(self.inching.get()),
            bool(self.arc.get()),
            bool(self.nonzero_unlock.get()),
        ) + raw_values

    def send(self, confirm_arc=True):
        try:
            values = self.command_values()
        except ValueError as error:
            self.set_result(False, str(error))
            return
        if confirm_arc and values[6] and not messagebox.askyesno(
            "Confirm ARC output",
            "ARC command is ON. Send this command to the H600 register image?",
        ):
            self.arc.set(False)
            self.set_result(False, "ARC command cancelled; forcing ARC OFF")
            self.send(confirm_arc=False)
            return
        self.command_result.configure(text="Sending…")
        threading.Thread(
            target=self.node.send_command,
            args=(values,),
            daemon=True,
        ).start()

    def control_toggled(self, selected):
        # Forward and reverse wire feed must never be commanded together.
        if selected == "reverse" and self.reverse_inching.get():
            self.inching.set(False)
        elif selected == "forward" and self.inching.get():
            self.reverse_inching.set(False)
        self.send()

    def force_off(self):
        self.robot_ready.set(False)
        self.robot_error.set(False)
        self.touch.set(False)
        self.gas.set(False)
        self.reverse_inching.set(False)
        self.inching.set(False)
        self.arc.set(False)
        self.current_raw.set(0)
        self.voltage_raw.set(0)
        self.v_offset_raw.set(0)
        self.nonzero_unlock.set(False)
        self.send()

    def control_server(self, start):
        host = self.bind_host.get().strip() or "0.0.0.0"
        self.server_result.configure(text="Starting…" if start else "Stopping…")
        threading.Thread(
            target=self.node.set_server,
            args=(start, host, H600_PORT),
            daemon=True,
        ).start()

    def set_server_result(self, success, message):
        self.server_result.configure(
            text=message,
            foreground="#137333" if success else "#b3261e",
        )

    @staticmethod
    def register_meaning(address):
        meanings = {
            201: "Command: robot ready (bit0)",
            202: "Command: error/touch/gas/reverse inch/inch/ARC (b7/b4/b3/b2/b1/b0)",
            204: "Command: weld current setpoint",
            205: "Command: weld voltage setpoint",
            206: "Command: voltage offset",
            211: "Status: heartbeat/error/welding/touch/info",
            212: "Status: weld current feedback",
            213: "Status: weld voltage feedback",
            216: "Status: single candidate",
        }
        return meanings.get(address, "Holding register")

    def refresh_registers(self):
        try:
            start = int(self.register_start.get())
            end = int(self.register_end.get())
        except (ValueError, tk.TclError):
            self.set_register_result(False, "Addresses must be integers")
            return
        count = end - start + 1
        if start < 0 or end > 65535 or count < 1 or count > 1000:
            self.set_register_result(False, "Select 1..1000 registers in 0..65535")
            return
        threading.Thread(
            target=self.node.get_registers,
            args=(start, count),
            daemon=True,
        ).start()

    def auto_refresh_registers(self):
        if not self._closing and self.register_auto.get():
            self.refresh_registers()
        if not self._closing:
            self.root.after(500, self.auto_refresh_registers)

    def update_registers(self, success, message, start, values):
        if not success:
            self.set_register_result(False, message)
            return
        self.register_table.delete(*self.register_table.get_children())
        nonzero_only = self.register_nonzero.get()
        changed_count = 0
        for offset, value in enumerate(values):
            address = start + offset
            old = self.previous_registers.get(address)
            changed = old is not None and old != value
            self.previous_registers[address] = value
            if nonzero_only and value == 0:
                continue
            if changed:
                changed_count += 1
            self.register_table.insert(
                "",
                tk.END,
                values=(address, value, f"0x{value:04X}", self.register_meaning(address)),
                tags=("changed",) if changed else (),
            )
        self.register_result.configure(text=f"{message} · {changed_count} changed")

    def set_register_result(self, success, message):
        self.register_result.configure(
            text=message,
            foreground="#137333" if success else "#b3261e",
        )

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
