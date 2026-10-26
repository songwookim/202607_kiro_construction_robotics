"""Persistent journal for Scan + 4 Passes, independent of GUI and ARC sessions."""

from datetime import datetime
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from uuid import uuid4

from construct_robot.io.teaching_yaml import atomic_text_writer


class ScanCycleLog:
    """One unique execution directory; append and flush events as work completes.

    summary.json is an atomic snapshot. events.jsonl remains readable after an
    interrupted process; a RUNNING summary means no normal finish was recorded.
    The executor handles write errors through its existing failure cleanup.
    """

    def __init__(self, root, requested_steps, conditions, fake_arc):
        root = Path(os.environ.get("KIRO_SCAN_CYCLE_LOG_ROOT", root)).expanduser()
        if root.name == "scan_four_pass_cycles":
            root = root.parent
        self.cycle_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]
        self.directory = root / f"scan_four_pass_{self.cycle_id}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self._feedback_root = root
        self._latest_pointer = root / "latest_scan_cycle.txt"
        # The initial scan determines the existing four-pass result folder.
        # Bind after teaching generation so its history archive stays small.
        self.bundle_directory = None
        self._bundle_files = {}
        self._unavailable_artifacts = []
        self._lock = threading.RLock()
        self._started = time.monotonic()
        self._closed = False
        self.document = {
            "schema": "construct_robot_scan_weld_cycle_log_v1",
            "cycle_id": self.cycle_id,
            "started": self._now(), "status": "RUNNING", "fake_arc": bool(fake_arc),
            "requested_steps": requested_steps, "execution_conditions": conditions,
            "steps": [], "results": [], "artifacts": [], "event_count": 0,
            "bundle_directory": None,
        }
        self.event("cycle_start", fake_arc=bool(fake_arc))
        self._save_summary()
        with atomic_text_writer(self._latest_pointer) as stream:
            stream.write(str(self.directory) + "\n")

    @staticmethod
    def _now():
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    def _save_summary(self):
        with atomic_text_writer(self.directory / "summary.json") as stream:
            json.dump(self.document, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if self.bundle_directory is None:
            return
        self._copy_to_bundle(self.directory / "summary.json", "summary.json", kind="cycle_log")
        # The journal lives in the result folder. Refresh append-only hashes
        # only at snapshot points, not for every motion/timing event.
        for name in ("events.jsonl", "cycle.log"):
            path = self.bundle_directory / name
            if path.is_file():
                self._bundle_files[name] = dict(file=name, source=str(self.directory / name), kind="cycle_log",
                                                bytes=path.stat().st_size, sha256=self._sha256(path))
        with atomic_text_writer(self.bundle_directory / "manifest.json") as stream:
            json.dump(dict(cycle=self.cycle_id, status=self.document["status"],
                           fake_arc=self.document["fake_arc"], original_cycle_directory=str(self.directory),
                           files=list(self._bundle_files.values()),
                           unavailable_artifacts=self._unavailable_artifacts), stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _copy_to_bundle(self, source, name, **metadata):
        """Publish an independent verified copy; never move or alter the source artifact."""
        target = self.bundle_directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() == target.resolve():
            self._bundle_files[name] = dict(file=name, source=str(source), bytes=source.stat().st_size,
                                            sha256=self._sha256(source), **metadata)
            return
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        before = source.stat()
        try:
            shutil.copyfile(source, temporary)
            digest = self._sha256(temporary)
            after = source.stat()
            if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                    or digest != self._sha256(source)):
                raise OSError(f"scan bundle source changed while copying: {source}")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        self._bundle_files[name] = dict(file=name, source=str(source), bytes=target.stat().st_size,
                                        sha256=digest, **metadata)

    def _attach_bundle(self, directory=None):
        if self.bundle_directory is not None:
            return
        if directory is None:
            initial = next((a for a in self.document["artifacts"]
                            if a.get("kind") in ("initial_scan", "precision_sensing_result") and a.get("path")), None)
            source = Path(initial["path"]) if initial else None
            name = (source.parent.name if source.suffix else source.name) if source else None
            directory = self._feedback_root / f"scan_four_pass_{name}" if name else self.directory
        directory = Path(directory)
        manifest = directory / "manifest.json"
        if manifest.is_file() and json.loads(manifest.read_text()).get("cycle") != self.cycle_id:
            raise ValueError(f"four-pass folder already belongs to another execution: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        previous = self.directory
        if previous.resolve() != directory.resolve():
            # Promote the journal into the teaching folder. Check every target
            # before moving anything; a different execution must remain intact.
            files = list(previous.iterdir())
            for source in files:
                target = directory / source.name
                if target.exists():
                    raise ValueError(f"refusing to replace an existing cycle log: {target}")
            moved = []
            try:
                for source in files:
                    target = directory / source.name
                    source.rename(target)
                    moved.append((source, target))
                previous.rmdir()
            except OSError:
                previous.mkdir(exist_ok=True)
                for source, target in reversed(moved):
                    target.rename(source)
                raise
            self.directory = directory
        self.bundle_directory = directory
        self.document["bundle_directory"] = str(directory)
        self.document["execution_conditions"]["scan_cycle_log_directory"] = str(directory)
        with atomic_text_writer(self._latest_pointer) as stream:
            stream.write(str(directory) + "\n")
        for name in ("events.jsonl", "cycle.log"):
            if (self.directory / name).is_file():
                self._copy_to_bundle(self.directory / name, name, kind="cycle_log")
        for artifact in self.document["artifacts"]:
            self._collect_artifact(artifact)

    def _collect_artifact(self, artifact):
        if self.bundle_directory is None:
            return
        source = Path(artifact.get("path") or "").expanduser()
        kind, number = artifact.get("kind"), artifact.get("pass_number")
        if kind not in ("initial_scan", "precision_sensing_result", "post_weld_scan",
                        "weld_feedback", "four_pass_teaching", "saved_four_pass_teaching", "post_weld_ply"):
            return
        if not artifact.get("path") or not source.exists():
            self._unavailable_artifacts.append(copy.deepcopy(artifact))
            return
        if kind in ("four_pass_teaching", "saved_four_pass_teaching"):
            files = [(p, p.name) for p in sorted(source.glob("pass_[1-4].yaml"))]
        elif kind == "weld_feedback":
            files = [(source, f"pass_{int(number)}_weld_feedback.log")]
        elif kind == "post_weld_ply":
            files = [(source, f"npz/pass_{int(number)}_rescan.ply")]
        else:
            prefix = "npz/" + ("initial_scan" if kind in ("initial_scan", "precision_sensing_result") else f"pass_{int(number)}_rescan")
            if source.is_file():
                files = [(source, prefix + source.suffix)]
                result = source.parent / "precision_sensing_result.yaml"
                if result.is_file():
                    files.append((result, prefix + "_result.yaml"))
            else:
                files = [(p, prefix + (p.suffix if p.stem == "scan" else "_" + p.name))
                         for p in sorted(source.iterdir()) if p.is_file() and p.suffix in (".npz", ".json", ".yaml")]
        for path, name in files:
            self._copy_to_bundle(path, name, kind=kind, pass_number=number)

    def event(self, event, **data):
        with self._lock:
            if self._closed and event != "artifact":
                return
            # Artifact events are sent only after the producer closes its
            # files. Late background feedback still belongs to this cycle.
            if event == "artifact":
                source = Path(data.get("path") or "")
                if data.get("kind") == "saved_four_pass_teaching" and source.is_dir():
                    self._attach_bundle(self.directory)
                elif data.get("kind") == "four_pass_teaching" and source.is_dir():
                    self._attach_bundle(source)
                elif data.get("kind") == "precision_sensing_result" and source.is_file():
                    self._attach_bundle(self._feedback_root / f"scan_four_pass_{source.parent.name}")
                self._collect_artifact(data)
            entry = dict(timestamp=self._now(), elapsed_s=time.monotonic() - self._started,
                         event=event, **data)
            text = json.dumps(entry, ensure_ascii=False, allow_nan=False)
            # One writer for simultaneous ARC/motion rows. Flush before
            # returning, so a later failure does not erase completed steps.
            for name, line in (("events.jsonl", text), ("cycle.log", f"{entry['timestamp']} {event} {json.dumps(data, ensure_ascii=False)}")):
                with (self.directory / name).open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            self.document["event_count"] += 1
            if event == "step_end":
                self.document["results"].append(entry)
            if event == "artifact":
                self.document["artifacts"].append(data)
                self._save_summary()

    def prepared(self, steps):
        with self._lock:
            self.document["steps"] = copy.deepcopy(steps)
            for step in self.document["steps"]:
                step["sequence_number"] += 1  # first scan is cycle step 1
            self._save_summary()
            self.event("paths_prepared", step_count=len(steps))

    def finish(self, status, message):
        with self._lock:
            if self._closed:
                return
            self._attach_bundle()
            self.event("cycle_end", status=status, message=str(message))
            completed = {r["sequence_number"] for r in self.document["results"] if "sequence_number" in r}
            self.document["not_run_steps"] = [step for step in (self.document["steps"] or self.document["requested_steps"])
                                              if step["sequence_number"] not in completed]
            self.document.update(status=status, message=str(message), ended=self._now(),
                                 elapsed_seconds=time.monotonic() - self._started)
            self._save_summary()
            self._closed = True


def record_cycle_event(host, event, **data):
    """Return False on a journal error; the executor stops through normal cleanup."""
    journal = getattr(host, "_scan_cycle_journal", None)
    if journal is None:
        return True
    try:
        journal.event(event, **data)
        return not bool(getattr(host, "_scan_cycle_log_error", None))
    except (OSError, TypeError, ValueError) as error:
        message = f"Scan + 4 Passes log write failed: {error}"
        host._scan_cycle_log_error = message
        host.post(host.error, message)
        return False
