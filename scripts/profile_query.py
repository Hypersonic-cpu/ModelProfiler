"""Example orchestration: capture -> short OpenAI-compatible request -> stop."""

import argparse
import json
import urllib.request

from modelprofiler.nsys import NsysSession


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Say hello.")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=int, help="optional collection time limit")
    args = parser.parse_args()
    payload = json.dumps({
        "model": args.model, "prompt": args.prompt,
        "max_tokens": args.max_tokens, "temperature": 0, "stream": False,
    }).encode()
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/completions", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with NsysSession(args.session).capture(args.output, duration=args.duration) as report:
        print(f"READY {report}", flush=True)
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            print(response.read().decode())
    print(f"Capture stopped: {report}")


if __name__ == "__main__":
    main()
