"""Private detached duration controller. Failures go to the session .log file."""

import sys
import time

from .session import NsysSession


if __name__ == "__main__":
    name, nsys, token, deadline = sys.argv[1:]
    time.sleep(max(0, float(deadline) - time.monotonic()))
    NsysSession(name, nsys=nsys)._expire(token)
