"""Control an explicitly named, pre-instrumented nsys session."""

from contextlib import contextmanager
import fcntl
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid


class NsysError(RuntimeError):
    """An nsys command failed; its diagnostics are included in the message."""


class NsysSession:
    """Connect to a named session, never an arbitrary uninstrumented PID.

    Use the same nsys installation and OS user for launch and control.
    stop() ends collection only; it never shuts down the session or server.
    """

    def __init__(self, name: str, *, nsys: str | None = None):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
            raise ValueError("session name must match [A-Za-z][A-Za-z0-9_-]*")
        self.name = name
        executable = nsys or os.environ.get("NSYS_BIN", "nsys")
        self.nsys = shutil.which(executable)
        if self.nsys is None:
            raise NsysError(f"nsys executable not found: {executable}; set NSYS_BIN")
        identity = hashlib.sha256(f"{self.nsys}:{name}".encode()).hexdigest()[:24]
        directory = Path(tempfile.gettempdir()) / f"modelprofiler-nsys-{os.getuid()}"
        directory.mkdir(mode=0o700, exist_ok=True)
        self._control = directory / identity

    @contextmanager
    def _lock(self):
        # Serialize controllers, including detached duration watchdogs.
        with self._control.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _run(self, *args: str) -> str:
        result = subprocess.run(
            [self.nsys, *args], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        if result.returncode:
            raise NsysError(
                f"nsys {' '.join(args)} failed ({result.returncode}):\n{result.stdout}"
            )
        return result.stdout

    def launch(self, command: list[str], *, trace: str = "cuda,nvtx,osrt") -> str:
        """One-time server launch with tracing configured but collection off.

        Child workers inherit instrumentation. Service readiness is the caller's
        responsibility; successful launch does not mean model loading is complete.
        """
        if not command:
            raise ValueError("server command is required")
        # Launch stays in the foreground for the lifetime of the server. Inherit
        # logs rather than buffering a persistent service's output in memory.
        args = [
            self.nsys,
            "launch", f"--session-new={self.name}", f"--trace={trace}",
            "--cuda-graph-trace=node",
            *command,
        ]
        result = subprocess.run(args, check=False)
        if result.returncode:
            raise NsysError(f"nsys launch failed ({result.returncode}); see server log")
        return ""

    def status(self) -> str:
        return self._run("status", f"--session={self.name}")

    def _state(self) -> str:
        # In 2024.6, launched status also contains the word "collecting" in
        # its instructions; match launched before looking for collecting.
        status = self.status().lower()
        for state in ("launched", "processing", "stopped", "idle", "collecting"):
            if re.search(rf"\b{state}\b", status):
                return state
        raise NsysError(f"Unrecognized nsys status for {self.name}: {status}")

    def start(self, output: str | Path, *, duration: int | None = None,
              ready_timeout: float = 30) -> Path:
        """Return only after nsys reports collecting; duration is in seconds."""
        with self._lock():
            return self._start(output, duration=duration, ready_timeout=ready_timeout)

    def _start(self, output, *, duration, ready_timeout):
        if duration is not None and (isinstance(duration, bool)
                                     or not isinstance(duration, int) or duration <= 0):
            raise ValueError("duration must be a positive integer number of seconds")
        if not math.isfinite(ready_timeout) or ready_timeout <= 0:
            raise ValueError("ready_timeout must be positive and finite")
        if self._state() not in ("launched", "stopped", "idle"):
            raise NsysError(f"Session {self.name} is not ready for a new collection")
        prefix = Path(output).expanduser().absolute()
        if prefix.suffix == ".nsys-rep":
            prefix = prefix.with_suffix("")
        # nsys expands percent placeholders; reject them so returned path is exact.
        if "%" in str(prefix):
            raise ValueError("output path must not contain nsys '%' placeholders")
        report = Path(str(prefix) + ".nsys-rep")
        if report.exists() or Path(str(prefix) + ".qdstrm").exists():
            raise FileExistsError(f"capture output already exists: {prefix}")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        args = ["start", f"--session={self.name}", f"--output={prefix}",
                "--sample=none", "--cpuctxsw=none"]
        self._run(*args)
        token = uuid.uuid4().hex
        expires = time.monotonic() + duration if duration is not None else None
        deadline = time.monotonic() + ready_timeout
        try:
            self._control.write_text(token)
            while True:
                state = self._state()
                if state == "collecting":
                    if expires is not None:
                        self._schedule_stop(token, expires)
                    return report
                if state == "stopped" or time.monotonic() >= deadline:
                    raise NsysError(f"Collection did not become ready: {self.name} ({state})")
                time.sleep(0.05)
        except BaseException:
            # A failed readiness check must not leave a collection running.
            self._stop()
            raise

    def _schedule_stop(self, token: str, deadline: float) -> None:
        # nsys start (2024.6) has no --duration option. A detached watchdog
        # survives the CLI process and only stops its own capture generation.
        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
        with self._control.with_suffix(".log").open("a") as log:
            watchdog = subprocess.Popen(
                [sys.executable, "-m", "modelprofiler.nsys._watchdog", self.name,
                 self.nsys, token, str(deadline)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=env,
            )
        # Reap it if the API caller stays alive; a daemon thread does not hold
        # the CLI open and the detached child survives the CLI's exit.
        threading.Thread(target=watchdog.wait, daemon=True).start()

    def _expire(self, token: str) -> None:
        with self._lock():
            if self._control.exists() and self._control.read_text() == token:
                self._stop()

    def stop(self) -> str:
        """Stop and finalize a running capture; tolerate an elapsed duration."""
        with self._lock():
            return self._stop()

    def _stop(self) -> str:
        if self._state() == "collecting":
            result = self._run("stop", f"--session={self.name}")
        else:
            result = self.status()
        self._control.unlink(missing_ok=True)
        return result

    @contextmanager
    def capture(self, output: str | Path, *, duration: int | None = None):
        """Capture caller-owned work and finalize even if that work raises."""
        report = self.start(output, duration=duration)
        try:
            yield report
        finally:
            self.stop()
