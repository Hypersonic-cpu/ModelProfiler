"""CPU-only lifecycle integration: NSYS_INTEGRATION_BIN=/path/to/nsys python -m unittest discover -s tests -v."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid

from modelprofiler.nsys import NsysError, NsysSession


@unittest.skipUnless(os.environ.get("NSYS_INTEGRATION_BIN"), "set NSYS_INTEGRATION_BIN")
class LifecycleTest(unittest.TestCase):
    def test_persistent_process_lifecycle(self):
        binary = os.environ["NSYS_INTEGRATION_BIN"]
        session = NsysSession("mp-test-" + uuid.uuid4().hex, nsys=binary)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Only this test-owned CPU process is launched and eventually killed.
            with (root / "launch.log").open("w") as log:
                process = subprocess.Popen(
                    [binary, "launch", f"--session-new={session.name}",
                     "--trace=osrt", "/bin/sleep", "120"],
                    stdout=log, stderr=log, start_new_session=True,
                )
                try:
                    deadline = time.monotonic() + 15
                    while True:
                        try:
                            if session._state() == "launched":
                                break
                        except NsysError:
                            pass
                        if time.monotonic() > deadline:
                            self.fail("session launch timed out")
                        time.sleep(0.05)

                    # Caller work begins only after READY. Failed work still
                    # finalizes the report and leaves the target alive.
                    with self.assertRaisesRegex(RuntimeError, "request failed"):
                        with session.capture(root / "failed") as report:
                            self.assertEqual(session._state(), "collecting")
                            with self.assertRaises(NsysError):
                                session.start(root / "overlap")
                            self.assertEqual(session._state(), "collecting")
                            raise RuntimeError("request failed")
                    self.assertTrue(report.is_file())
                    with self.assertRaises(FileExistsError):
                        session.start(root / "failed")
                    self.assertEqual(session._state(), "launched")
                    self.assertIsNone(process.poll())

                    # A timer from an early-stopped capture cannot stop a newer one.
                    session.start(root / "early", duration=2)
                    token = session._control.read_text()
                    session.stop()
                    next_report = session.start(root / "next")
                    session._expire(token)
                    self.assertEqual(session._state(), "collecting")
                    session.stop()
                    self.assertTrue(next_report.is_file())

                    # Test the CLI process exiting while its duration watchdog
                    # remains alive and finalizes independently.
                    result = subprocess.run(
                        [sys.executable, "-m", "modelprofiler.nsys", "--nsys", binary,
                         "--session", session.name, "start", "--output", str(root / "timed"),
                         "--duration", "1"],
                        capture_output=True, text=True, check=True,
                    )
                    self.assertIn("READY", result.stdout)
                    deadline = time.monotonic() + 15
                    while not (root / "timed.nsys-rep").is_file():
                        if time.monotonic() > deadline:
                            self.fail("watchdog did not finalize report")
                        time.sleep(0.1)
                    # stop waits for a watchdog currently finalizing under lock.
                    session.stop()
                    self.assertEqual(session._state(), "launched")
                    self.assertIsNone(process.poll())

                    # Exercise the actual query script against a CPU HTTP stub;
                    # the handler observes that tracing has started before POST.
                    observed = []

                    class Handler(BaseHTTPRequestHandler):
                        def do_POST(self):
                            observed.append(session._state())
                            self.rfile.read(int(self.headers["Content-Length"]))
                            self.send_response(200)
                            self.end_headers()
                            self.wfile.write(b'{"choices":[{"text":"hello"}]}')

                        def log_message(self, *args):
                            pass

                    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as http:
                        thread = threading.Thread(target=http.serve_forever, daemon=True)
                        thread.start()
                        try:
                            subprocess.run(
                                [sys.executable, "-m", "scripts.profile_query", "--session",
                                 session.name, "--model", "test", "--output", str(root / "query"),
                                 "--base-url", f"http://127.0.0.1:{http.server_port}/v1"],
                                env={**os.environ, "NSYS_BIN": binary},
                                capture_output=True, text=True, check=True,
                            )
                        finally:
                            http.shutdown()
                            thread.join()
                    self.assertEqual(observed, ["collecting"])
                    self.assertTrue((root / "query.nsys-rep").is_file())
                    self.assertEqual(session._state(), "launched")
                    self.assertIsNone(process.poll())
                finally:
                    subprocess.run(
                        [binary, "shutdown", f"--session={session.name}", "--kill=sigterm"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    process.wait(timeout=15)


if __name__ == "__main__":
    unittest.main()
