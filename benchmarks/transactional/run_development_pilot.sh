#!/usr/bin/env bash
# Shared batch runner: nine standard runs, or three full/upstream runs.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

pilot_profiles=(baseline reducer full)
pilot_prefix=development-pilot
if [[ "${1:-}" == "--full-upstream" ]]; then
  pilot_profiles=(full)
  pilot_prefix=full-upstream
  shift
fi
if (( $# > 1 )); then
  printf 'Usage: bash %s [--full-upstream] [model]\n' "$0" >&2
  exit 2
fi

# Direct access was verified for the development provider on this WSL host.
# Only this process tree is affected; system proxy settings are unchanged.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset FAST_AGENT_WEBDEBUG
export UV_CACHE_DIR=/tmp/fast-agent-uv-cache

pilot_model="${1:-aliyun.qwen3.8-max}"
pilot_commit="$(git rev-parse HEAD)"
mkdir -p .txagent-runs
pilot_root="$(mktemp -d "$project_root/.txagent-runs/$pilot_prefix-${pilot_commit:0:8}.XXXXXX")"
printf '本次结果目录：%s\n' "$pilot_root"
printf '%s\n' "$pilot_commit" > "$pilot_root/implementation_commit.txt"
git status --short > "$pilot_root/worktree_status.txt"
sha256sum benchmarks/transactional/tasks/*.yaml > "$pilot_root/manifest_checksums.txt"

for task in boundary-check-001 long-output-001 recovery-path-001; do
  for profile in "${pilot_profiles[@]}"; do
    if [[ "$(git rev-parse HEAD)" != "$pilot_commit" ]]; then
      printf '实现提交已变化，停止本批实验。\n' >&2
      exit 1
    fi
    strategy=semantic
    if [[ "$profile" == baseline || "$pilot_prefix" == full-upstream ]]; then
      strategy=upstream
    fi
    run_dir="$pilot_root/$task-$profile-$strategy"
    printf '\n开始：%s / %s + %s\n' "$task" "$profile" "$strategy"
    if uv run benchmarks/transactional/run_development_pilot.py \
      --manifest "benchmarks/transactional/tasks/$task.yaml" \
      --profile "$profile" --tool-output-strategy "$strategy" \
      --model "$pilot_model" --output "$run_dir" \
      2>&1 | tee "$run_dir.log"; then
      run_rc=0
    else
      run_rc=$?
    fi
    printf '%s\t%s\t%s\t%s\n' "$task" "$profile" "$strategy" "$run_rc" \
      >> "$pilot_root/exit_codes.tsv"
    if (( run_rc >= 128 )); then
      exit "$run_rc"
    fi
    uv run python - "$run_dir/result.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("缺少 result.json，停止批量运行，请保留日志。")
result = json.loads(path.read_text())
for key in (
    "status", "tool_output_strategy", "semantic_reducer_version",
    "verification_passed", "llm_calls", "tool_calls", "input_tokens",
    "output_tokens", "recovery_attempts", "promotion_result", "failure_reason",
):
    print(f"{key}: {result.get(key)}")
if result.get("error_type") == "PilotProviderError":
    raise SystemExit("Provider 故障，停止后续调用，请保留日志。")
if result.get("status") == "cancelled":
    raise SystemExit("运行已取消，停止批量运行。")
PY
  done
done

printf '\n本批 %s 次开发运行结束，结果目录：%s\n' "$((3 * ${#pilot_profiles[@]}))" "$pilot_root"
