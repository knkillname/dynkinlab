#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Run the Colab runtime contract assertions in the current interpreter."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECK = r"""
import re
import sys

import matplotlib
import networkx
import numpy


def release(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value)[:2])


assert sys.version_info[:2] == (3, 13), sys.version
assert release(numpy.__version__) >= (2, 1), numpy.__version__
assert release(networkx.__version__) >= (3, 7), networkx.__version__
assert release(matplotlib.__version__) >= (3, 10), matplotlib.__version__

import dynkinlab

assert isinstance(dynkinlab.__version__, str)
assert dynkinlab.__version__

print(f"Python {sys.version.split()[0]}")
print(f"NumPy {numpy.__version__}")
print(f"NetworkX {networkx.__version__}")
print(f"matplotlib {matplotlib.__version__}")
print(f"dynkinlab {dynkinlab.__version__}")
"""


def run_check() -> int:
    """Run the runtime assertions with the current interpreter."""
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    print(f"colab-smoke: checking {sys.executable}")
    return subprocess.run(
        [sys.executable, "-c", CHECK],
        check=False,
        env=environment,
    ).returncode


def main() -> int:
    """Run the Colab runtime checks."""
    status = run_check()
    if status != 0:
        print("colab-smoke: the compatibility check failed.", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
