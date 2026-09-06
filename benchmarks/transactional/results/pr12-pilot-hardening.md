# PR12 development Pilot hardening

This report records development evidence gathered while hardening the
transactional Pilot runner, recovery path, and coding policy. It is a mechanism
smoke test and an exploratory comparison, not a frozen held-out benchmark.

The selected baseline and reducer runs are from commit `34c94799` in
`.txagent-runs/development-pilot-34c94799.6JlUhs/`. The selected full runs are
from the later `coding-v3` policy commit `6f93cd0b` in
`.txagent-runs/full-policy-v3-6f93cd0b.NAHeyO/`. Because the four groups were
not run from one implementation commit, comparisons across all four groups are
descriptive only.

## Results

All runs used `aliyun.qwen3.8-max`, the same task-specific external verification
command, and a maximum budget of 40 logical LLM calls, 80 tool calls, and 600
seconds. Provider attempts are reported separately from logical LLM calls.

| Task | Profile / strategy | Verified | LLM / provider calls | Tool calls | Input tokens | Output tokens | Total tokens | Model-visible tool bytes | Controlled bytes | Policy denials | Recovery | Wall time | Promotion |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| Boundary check | baseline / upstream | yes | 10 / 10 | 9 | 28,560 | 1,872 | 30,432 | 8,844 | — | — | 0 | 37.48s | — |
| Boundary check | reducer / semantic-v1 | yes | 10 / 10 | 9 | 30,842 | 1,804 | 32,646 | 9,602 | — | — | 0 | 262.73s | — |
| Boundary check | full / semantic-v1 | yes | 11 / 11 | 10 | 38,179 | 1,947 | 40,126 | 13,590 | — | 0 | 0 | 40.84s | applied |
| Boundary check | full / upstream | yes | 12 / 12 | 11 | 39,537 | 1,765 | 41,302 | 11,812 | — | 0 | 0 | 38.62s | applied |
| Long output | baseline / upstream | yes | 12 / 12 | 11 | 49,256 | 2,276 | 51,532 | 24,421 | 17,328 -> 17,328 | — | 0 | 49.95s | — |
| Long output | reducer / semantic-v1 | yes | 16 / 16 | 15 | 52,971 | 2,435 | 55,406 | 11,710 | 17,329 -> 1,271 | — | 0 | 54.05s | — |
| Long output | full / semantic-v1 | yes | 13 / 13 | 12 | 45,232 | 2,959 | 48,191 | 12,799 | 17,326 -> 1,269 | 1 | 0 | 60.77s | applied |
| Long output | full / upstream | yes | 12 / 12 | 11 | 64,645 | 2,829 | 67,474 | 29,480 | 17,325 -> 17,325 | 0 | 0 | 58.97s | applied |
| Recovery observation | baseline / upstream | yes | 9 / 9 | 8 | 25,979 | 2,968 | 28,947 | 7,902 | — | — | 0 | 57.86s | — |
| Recovery observation | reducer / semantic-v1 | yes | 13 / 13 | 12 | 44,329 | 4,914 | 49,243 | 14,961 | — | — | 0 | 95.75s | — |
| Recovery observation | full / semantic-v1 | yes | 11 / 11 | 10 | 38,483 | 4,083 | 42,566 | 11,916 | — | 0 | 0 | 79.91s | applied |
| Recovery observation | full / upstream | yes | 12 / 12 | 11 | 37,856 | 2,729 | 40,585 | 10,050 | — | 0 | 0 | 57.35s | applied |

| Profile / strategy | Runs verified | LLM calls | Tool calls | Total tokens | Model-visible tool bytes | Average wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline / upstream | 3/3 | 31 | 28 | 110,911 | 41,167 | 48.43s |
| reducer / semantic-v1 | 3/3 | 39 | 36 | 137,295 | 36,273 | 137.51s |
| full / semantic-v1 | 3/3 | 35 | 32 | 130,883 | 38,305 | 60.51s |
| full / upstream | 3/3 | 36 | 33 | 149,361 | 51,342 | 51.65s |

## What this establishes

- All 12 runs passed their task's fixed external verification command. All six
  full runs also recorded `run.verified` followed by `promotion.applied`.
- On the latest controlled long-output run, semantic-v1 reduced one tool result
  from 17,326 bytes to 1,269 bytes, about 92.7%. This is not the task's token
  reduction.
- On the latest same-commit full runs, semantic-v1 used 130,883 total tokens
  versus 149,361 with upstream output, about 12.4% fewer. One trajectory per
  configuration cannot establish causality or expected savings.
- The latest `coding-v3` full batch recorded one policy denial: a structured
  absolute path outside the worktree. It recorded no Shell command-text path
  false positives.
- No real-provider run naturally triggered recovery. Deterministic integration
  tests instead cover repeated timeout and nonzero-exit failures through real
  Harness/ToolRunner execution, rollback, Handoff consumption, a changed plan,
  verification, and promotion.

## Limits

- Every task/configuration pair ran once, and the baseline/reducer and full
  groups came from different implementation commits.
- These are small development fixtures on which all configurations succeeded;
  they do not measure success-rate improvement.
- Policy enforcement is application-level. Arbitrary local Shell commands still
  run with the host permissions of the fast-agent process; this is not an OS or
  container sandbox.
- PR13 must freeze one implementation commit, model configuration, budgets,
  three held-out tasks, four configurations, and three repetitions before any
  held-out result is inspected.
