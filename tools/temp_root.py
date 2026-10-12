#!/usr/bin/env python3
"""Run TorchTitan host tools with an owned, checked temporary root."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def temporary_root(value=None):
    selected = value if value is not None else os.environ.get("TMPDIR")
    root = Path(selected).expanduser() if selected else ROOT / ".scratch" / "tmp"
    if not root.is_absolute():
        raise ValueError("TMPDIR must be absolute")
    root = root.resolve()
    forbidden = Path("/") / "tmp"
    if root == forbidden or forbidden in root.parents:
        raise ValueError("TMPDIR must be outside the system temporary directory")
    if root == Path("/"):
        raise ValueError("TMPDIR must be an owned subdirectory")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_uid != os.getuid():
        raise ValueError("TMPDIR must be owned by the current user")
    if root.stat().st_mode & 0o022:
        raise ValueError("TMPDIR must not be writable by other users")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        root = temporary_root()
    except (ValueError, OSError) as error:
        parser.exit(2, f"temporary root: {error}\n")
    if not args.command:
        print(root)
        return 0
    command = args.command[1:] if args.command[0] == "--" else args.command
    environment = dict(os.environ, TMPDIR=str(root), TMP=str(root), TEMP=str(root))
    return subprocess.call(command, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
