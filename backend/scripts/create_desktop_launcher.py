#!/usr/bin/env python3
"""Create a macOS desktop launcher for the Echo Prompt Calibrator."""

from __future__ import annotations

import argparse
import stat
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a desktop launcher (.command) for Echo Prompt Calibrator."
    )
    parser.add_argument(
        "--name",
        default="Launch Echo Prompt Calibrator",
        help="Launcher file name without extension",
    )
    parser.add_argument(
        "--desktop-dir",
        default=str(Path.home() / "Desktop"),
        help="Desktop directory path",
    )
    parser.add_argument(
        "--workspace-dir",
        default=str(Path(__file__).resolve().parents[2]),
        help="Project workspace root (auto-detected by default)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    desktop_dir = Path(args.desktop_dir).expanduser().resolve()
    workspace_dir = Path(args.workspace_dir).expanduser().resolve()
    launch_script = workspace_dir / "backend" / "scripts" / "launch_echo_calibrator.py"

    if not launch_script.exists():
        raise SystemExit(f"Launch script not found: {launch_script}")

    desktop_dir.mkdir(parents=True, exist_ok=True)
    launcher_path = desktop_dir / f"{args.name}.command"

    content = f"""#!/bin/zsh
cd "{workspace_dir}" || exit 1
if command -v python3 >/dev/null 2>&1; then
  python3 "backend/scripts/launch_echo_calibrator.py"
else
  python "backend/scripts/launch_echo_calibrator.py"
fi
"""
    launcher_path.write_text(content, encoding="utf-8")

    current_mode = launcher_path.stat().st_mode
    launcher_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"Created desktop launcher: {launcher_path}")
    print("Double-click it to open the Echo Prompt Calibrator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
