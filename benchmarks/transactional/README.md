# Transactional development Pilot

These three fixtures are a development set. Runs diagnose mechanisms and runner
behavior; they do not estimate held-out task success or demonstrate that a real
model reliably changes its plan after recovery.

The comparison packages remain baseline + upstream tool output, reducer +
semantic v1, and full + semantic v1. Profile and output strategy are independent
product settings; these packages are the Pilot's experimental choice.

## Common limits

All three manifests now allow 40 logical LLM calls, 80 tool calls and 600 seconds
per Agent session. These are ceilings, not targets. The Pilot composes public
ToolRunner hooks with existing hooks to enforce the same admission rules in all
profiles, including baseline. Native transactional LLM/tool/wall/output/token
limits are disabled **only in the temporary Pilot configuration** to avoid a
second, profile-dependent experimental gate. Product defaults are unchanged.
Full retains its recovery cap of 2 and its mandatory Completion Verifier.

- LLM calls count admitted logical calls, including calls whose provider fails.
  Provider retries remain within a logical call and consume elapsed time.
- Tool calls count admitted planned calls, including calls Full's policy denies;
  they are not a count of successful effects. A batch exceeding the remaining
  allowance is rejected entirely before any of its tools execute.
- The wall limit covers `session.send`, including Full's verification/recovery
  and promotion. Fixture preparation and the external grader are outside it.
  The external grader has the manifest's separate verification timeout.
- The ordinary ToolRunner iteration limit is set above the LLM ceiling so it
  does not introduce another, hidden stopping rule.

Budget changes alter manifest hashes. Keep earlier runs as historical evidence;
do not pool their counts or outcomes with this protocol as an unchanged experiment.

## Results and failure evidence

Every started run saves `result.json` (schema version 2), including failures and
handled cancellation. It records the implementation commit and dirty flag,
manifest hash, resolved budget, logical calls, usage, event kinds and failure
reason. A dirty flag means the commit alone does not identify the tested code.
Hard process termination or disk failure cannot guarantee saving final metrics.
Invalid/unreadable manifests fail before an experiment is started.

`provider_attempts` is the usage accumulator's recorded attempt count. Requests
that fail without usage reporting can be absent from it, so it is not a complete
HTTP request counter. Unknown tokens remain null; they are not fabricated zeros.
The former Pilot's `llm_calls` field used this provider-attempt count.

For reducer/full, `events.json` preserves detailed tool/run events before Harness
closes its stores. The runtime directory retains SQLite events, artifacts and
Full's isolated worktree. Observations are collected outside message history so
an aborted turn does not erase completed tool-output measurements. If a denied
controlled command has no stored raw output, its pre-reducer bytes remain null.

On failure, `failure.json` also contains the error type and reason, and the CLI
exits nonzero. `verification_passed: null` means the external grader was not run;
it does not mean tests failed. Grader failures and timeouts save captured output
and return nonzero. `run.verified` is Full's isolated-worktree verification;
`promotion.applied` is write-back. External grading inspects the cloned source
workspace after the session, so these are separate observations.

## Local validation without a provider

The Runner integration tests use the real Harness/ToolRunner and local shell,
replacing only the LLM provider response with a script. They check common LLM,
tool and wall limits, provider errors, retained failure evidence, and completion
for all three profiles. The Handoff test repeats either a real bounded shell timeout or a nonzero exit,
checks rollback and Handoff in the next request, emits a different repair,
and requires verification and promotion of the same workspace version.

```bash
UV_CACHE_DIR=/tmp/fast-agent-uv-cache uv run pytest -vv \
  tests/integration/transactional/test_pilot_runner.py \
  tests/integration/transactional/test_harness_runtime.py::test_handoff_replans_through_tool_runner_then_verifies_and_promotes
```

The Handoff test proves the wired timeout and nonzero-exit recovery paths with
scripted decisions, not autonomous replanning quality. Native session-shell
nonzero-exit signatures ignore only the generated process ID footer when it
matches canonical result metadata. Actual output and exit codes remain part of
the signature; raw artifacts and diagnostic summaries retain the original ID.
This does not normalize arbitrary timestamps, paths or application-generated IDs.
Handoff `reverted_files` is currently unpopulated; the test checks actual file
contents and the recovered version instead.

Historical selected PR11 results and their limitations are in
[the PR11 Pilot report](results/pr11-development-pilot.md); no new real-provider
results are implied by these local tests.
