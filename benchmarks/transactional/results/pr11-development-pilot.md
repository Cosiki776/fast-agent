# PR11 development Pilot

This development Pilot exercised three transactional profiles with
`aliyun.qwen3.8-max` on three deterministic fixture tasks. It is a mechanism
smoke test, not a formal or statistically significant benchmark.

Tool-output handling was configured independently from the transactional
profile: baseline used `tool_output.strategy=upstream`; reducer and full used
`tool_output.strategy=semantic` with `semantic_reducer_version=v1`.

## Results

| Task | Profile | Verified | LLM calls | Tool calls | Input tokens | Output tokens | Model-visible tool bytes | Wall time |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Boundary check | baseline | yes | 9 | 8 | 23,627 | 1,223 | 7,874 | 59.6s |
| Boundary check | reducer | yes | 8 | 7 | 20,863 | 885 | 5,952 | 45.3s |
| Boundary check | full | yes | 6 | 5 | 13,305 | 688 | 4,179 | 35.4s |
| Long output (controlled) | baseline | yes | 10 | 9 | 40,850 | 1,198 | 22,937 | 63.6s |
| Long output (controlled) | reducer | yes | 9 | 8 | 24,667 | 1,294 | 8,835 | 60.5s |
| Long output (controlled) | full | yes | 10 | 9 | 30,176 | 1,500 | 8,721 | 79.1s |
| Recovery observation | baseline | yes | 8 | 7 | 18,927 | 2,329 | 4,840 | 78.2s |
| Recovery observation | reducer | yes | 8 | 7 | 19,881 | 1,697 | 5,297 | 56.3s |
| Recovery observation | full | yes | 12 | 11 | 34,818 | 2,358 | 9,032 | 99.9s |

All nine runs passed the same external verification command for their task. All
three full-profile runs also recorded `run.verified` followed by
`promotion.applied`.

For the controlled long-output command, baseline exposed all 17,329 serialized
`CallToolResult` bytes to the model. Reducer and full exposed 1,274 of roughly
17,328 pre-reducer bytes, a 92.6% reduction. This measurement starts at the
`CallToolResult` received by the transactional coordinator; it does not claim to
measure unlimited raw process output below the shell runtime's own output cap.

## Interpretation limits

- Each task/profile pair ran once, so token and wall-time differences include
  provider and trajectory variance.
- The tasks are small development fixtures. Baseline solved all three, leaving
  no success-rate headroom for full on this sample.
- The recovery-observation task did not naturally trigger recovery. Stable
  recovery-to-verification-to-promotion behavior is covered by deterministic
  integration tests instead of forcing a real model to fail.
- The long-output task is a controlled mechanism check: all profiles had to run
  the same unfiltered command before editing. PR12 held-out evaluation uses
  natural tool behavior and permits model-selected filtering.
- The selected results were gathered across iterative PR11 Pilot reruns after
  fixing issues exposed by earlier attempts. Failed and superseded runs are not
  included in this summary.

The machine-readable source for this table is
[`pr11-development-pilot.json`](pr11-development-pilot.json).
