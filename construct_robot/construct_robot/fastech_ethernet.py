"""Thread-safe Ethernet adapter for the Fastech Ezi-IO Plus-E library.

The vendor API exposes mixed I/O products as one bit space.  In particular,
an Ethernet-I8O8 uses bits 0..7 for physical inputs 0..7 and bits 8..15 for
physical outputs 0..7.  Callers of this module always use physical channel
numbers; the product-specific offset stays here.
"""

from dataclasses import dataclass
import importlib
import ipaddress
import os
from pathlib import Path
import sys
import threading


FASTECH_VENDOR_DIRECTORY = (
    "251216_Program_Plus-E Linux_Library_Python_Ver.1.0.5_64bit"
)

# Device type values from the supplied MOTION_DEFINE.py.
EZI_IO_IN16 = 150
EZI_IO_IN32 = 151
EZI_IO_I8O8 = 155
EZI_IO_I16O16 = 156
EZI_IO_OUT16 = 160
EZI_IO_OUT32 = 161


@dataclass(frozen=True)
class FastechIoSnapshot:
    inputs: tuple
    outputs: tuple
    raw_input: int
    raw_output: int
    latch: int
    trigger_status: int


def find_fastech_library_dir():
    """Locate the supplied vendor Python wrapper and ARM64 shared library."""
    override = os.environ.get("FASTECH_LIBRARY_DIR", "").strip()
    candidates = [Path(override)] if override else []
    source = Path(__file__).resolve()
    candidates.extend(
        parent / FASTECH_VENDOR_DIRECTORY / "Library"
        for parent in source.parents
    )
    for candidate in candidates:
        if (
            candidate.is_dir()
            and (candidate / "FAS_EziMOTIONPlusE.py").is_file()
            and (candidate / "libEziMOTIONPlusE.so").exists()
        ):
            return candidate
    raise RuntimeError(
        "Fastech vendor Library directory was not found; set "
        "FASTECH_LIBRARY_DIR if it was moved"
    )


def load_fastech_api():
    library_dir = find_fastech_library_dir()
    path = str(library_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    api = importlib.import_module("FAS_EziMOTIONPlusE")
    return_codes = importlib.import_module("ReturnCodes_Define")
    return api, int(return_codes.FMM_OK)


def _io_layout(device_type):
    layouts = {
        EZI_IO_IN16: (16, 0, 0),
        EZI_IO_IN32: (32, 0, 0),
        EZI_IO_I8O8: (8, 8, 8),
        EZI_IO_I16O16: (16, 16, 16),
        EZI_IO_OUT16: (0, 16, 0),
        EZI_IO_OUT32: (0, 32, 0),
    }
    if int(device_type) not in layouts:
        raise RuntimeError(f"Unsupported Fastech I/O device type: {device_type}")
    return layouts[int(device_type)]


class FastechEthernetClient:
    """Synchronous Fastech Ethernet client for worker threads."""

    def __init__(self, ip="192.168.0.3", board_id=0, api=None, ok_code=None):
        self.ip = str(ipaddress.ip_address(str(ip).strip()))
        self.board_id = int(board_id)
        if not 0 <= self.board_id <= 255:
            raise ValueError("Fastech board ID must be in 0..255")
        if api is None:
            api, loaded_ok_code = load_fastech_api()
            ok_code = loaded_ok_code if ok_code is None else ok_code
        self._api = api
        self._ok = 0 if ok_code is None else int(ok_code)
        self._lock = threading.Lock()
        self.connected = False
        self.device_type = None
        self.device_version = ""
        self.input_count = 0
        self.output_count = 0
        self.output_offset = 0

    def connect(self):
        octets = [int(part) for part in self.ip.split(".")]
        with self._lock:
            if self.connected:
                return self.device_type, self.device_version
            result = self._api.FAS_ConnectTCP(*octets, self.board_id)
            if int(result) == 0:
                raise RuntimeError(
                    f"Fastech TCP connection failed: {self.ip}, board {self.board_id}"
                )
            try:
                status, device_type, version = self._api.FAS_GetSlaveInfo(
                    self.board_id
                )
                self._require_ok(status, "FAS_GetSlaveInfo")
                input_count, output_count, output_offset = _io_layout(
                    device_type
                )
            except Exception:
                self._api.FAS_Close(self.board_id)
                raise
            self.device_type = int(device_type)
            self.device_version = str(version)
            self.input_count = input_count
            self.output_count = output_count
            self.output_offset = output_offset
            self.connected = True
            return self.device_type, self.device_version

    def close(self):
        with self._lock:
            if self.connected:
                self._api.FAS_Close(self.board_id)
            self.connected = False

    def _require_ok(self, status, operation):
        if int(status) != self._ok:
            raise RuntimeError(f"{operation} failed with code {int(status)}")

    def _require_connected(self):
        if not self.connected:
            raise RuntimeError("Fastech I/O is not connected")

    def read_io(self):
        with self._lock:
            self._require_connected()
            input_status, raw_input, latch = self._api.FAS_GetInput(
                self.board_id
            )
            self._require_ok(input_status, "FAS_GetInput")
            output_status, raw_output, trigger_status = self._api.FAS_GetOutput(
                self.board_id
            )
            self._require_ok(output_status, "FAS_GetOutput")
            return self._snapshot(raw_input, raw_output, latch, trigger_status)

    def _snapshot(self, raw_input, raw_output, latch=0, trigger_status=0):
        raw_input = int(raw_input)
        raw_output = int(raw_output)
        return FastechIoSnapshot(
            inputs=tuple(
                bool(raw_input & (1 << channel))
                for channel in range(self.input_count)
            ),
            outputs=tuple(
                bool(raw_output & (1 << (self.output_offset + channel)))
                for channel in range(self.output_count)
            ),
            raw_input=raw_input,
            raw_output=raw_output,
            latch=int(latch),
            trigger_status=int(trigger_status),
        )

    def set_output(self, channel, enabled):
        return self.set_outputs({int(channel): bool(enabled)})

    def set_outputs(self, values):
        set_mask = 0
        clear_mask = 0
        for channel, enabled in values.items():
            channel = int(channel)
            if not 0 <= channel < self.output_count:
                raise ValueError(
                    f"Fastech physical output channel must be in "
                    f"0..{self.output_count - 1}"
                )
            mask = 1 << (self.output_offset + channel)
            if enabled:
                set_mask |= mask
            else:
                clear_mask |= mask
        if set_mask & clear_mask:
            raise ValueError("The same Fastech output cannot be set and cleared")
        with self._lock:
            self._require_connected()
            status = self._api.FAS_SetOutput(
                self.board_id, set_mask, clear_mask
            )
            self._require_ok(status, "FAS_SetOutput")
            output_status, raw_output, trigger_status = self._api.FAS_GetOutput(
                self.board_id
            )
            self._require_ok(output_status, "FAS_GetOutput")
            input_status, raw_input, latch = self._api.FAS_GetInput(
                self.board_id
            )
            self._require_ok(input_status, "FAS_GetInput")
            snapshot = self._snapshot(
                raw_input, raw_output, latch, trigger_status
            )
            mismatches = [
                channel
                for channel, enabled in values.items()
                if snapshot.outputs[int(channel)] != bool(enabled)
            ]
            if mismatches:
                raise RuntimeError(
                    "Fastech output readback mismatch: "
                    + ", ".join(f"DO{channel}" for channel in mismatches)
                )
            return snapshot
