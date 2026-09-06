#!/usr/bin/env bash
# Supplementary full/upstream entry point; shares logging and failure handling.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_development_pilot.sh" --full-upstream "$@"
