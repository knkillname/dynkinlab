#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# Create the Hatch environments used by the dev container and expose a stable
# interpreter path for VS Code.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

hatch env create
hatch env create docs
hatch env create hatch-test.py3.13

# Hatch's system environment stores a marker here; create the directory so the
# Colab runtime check can use the container's Python without installing anything.
mkdir -p "$HOME/.local/share/hatch/env/system"
hatch env create colab

# VS Code points at this stable path instead of a Hatch hash directory.
interpreter_dir="$HOME/.local/share/dynkinlab"
mkdir -p "$interpreter_dir"
ln -sfn "$(hatch env find hatch-test.py3.13)" "$interpreter_dir/test-venv"

printf '\nOpenCode: %s\n' "$(opencode --version)"
printf 'OpenSpec: %s\n' "$(openspec --version)"
printf 'Dev container ready. Run hatch run verify.\n'
