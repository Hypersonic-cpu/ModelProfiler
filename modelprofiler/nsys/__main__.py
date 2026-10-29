import argparse

from .session import NsysError, NsysSession


def main() -> None:
    parser = argparse.ArgumentParser(description="Control a pre-instrumented nsys session")
    parser.add_argument("--nsys", help="nsys executable (default: NSYS_BIN or PATH)")
    parser.add_argument("--session", required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    launch = sub.add_parser("launch", help="one-time instrumented server launch")
    launch.add_argument("--trace", default="cuda,nvtx,osrt")
    launch.add_argument("command", nargs=argparse.REMAINDER)
    start = sub.add_parser("start", help="wait for collecting, then print READY")
    start.add_argument("--output", required=True)
    start.add_argument("--duration", type=int)
    sub.add_parser("stop", help="stop capture without stopping the server")
    sub.add_parser("status")
    args = parser.parse_args()
    try:
        session = NsysSession(args.session, nsys=args.nsys)
        if args.action == "launch":
            command = args.command
            if command[:1] == ["--"]:
                command = command[1:]
            print(session.launch(command, trace=args.trace), end="")
        elif args.action == "start":
            report = session.start(args.output, duration=args.duration)
            print(f"READY {report}", flush=True)
        else:
            print(getattr(session, args.action)(), end="")
    except (NsysError, ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
