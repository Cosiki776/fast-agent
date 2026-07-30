# Transactional Local Tool Execution

- Status: Accepted
- Date: 2026-07-30
- Baseline: `b8f1c32898ef`

## Context

fast-agent already has a complete tool loop. `ToolRunner.generate_tool_call_response()`
invokes `before_tool_call` once, passes the complete assistant message to
`McpAgent.run_tools()`, and invokes `after_tool_call` with the aggregated tool-result
message.

This is a batch boundary. One assistant message can contain multiple tool calls, so
the existing hooks cannot independently assign transaction IDs, persist intent,
apply policy, checkpoint the workspace, or record the outcome of each side effect.
Changing those hooks to per-call hooks would also change their existing contract and
affect every agent using the tool loop.

Within `McpAgent`, the current execution path is:

```text
McpAgent.run_tools()
  -> _plan_mcp_tool_calls()
  -> _run_parallel_planned_tool_calls()
     or _run_sequential_planned_tool_calls()
  -> _execute_mcp_planned_tool_call()
  -> call_tool()
  -> _call_local_tool()
     or MCPAggregator.call_tool()
```

`_execute_mcp_planned_tool_call()` receives one planned call with its correlation ID,
resolved execution name, arguments, and routing metadata. It is therefore the
narrowest existing boundary that can govern one call without replacing planning,
result aggregation, display, timing, permission handling, or the surrounding agent
loop.

## Decision

### Scope

The first phase adds an opt-in transactional execution path for a single agent,
repository, and run. It supports only local coding capabilities exposed through
`McpAgent`:

- `read_text_file`;
- `write_text_file`;
- `apply_patch`;
- the local shell/bash tool.

Remote MCP tools, child-agent tools, human-input tools, and other built-in tools are
not transactionally governed in this phase.

### Per-call boundary

The optional execution interceptor will wrap the call to `McpAgent.call_tool()` made
by `_execute_mcp_planned_tool_call()`. Eligibility will be determined from explicit
planned-call and local-runtime information before invoking the interceptor; it will
not infer tool kinds from arbitrary remote tool names.

The interceptor contract will:

- receive a typed request containing the run ID, tool-call correlation ID, tool name,
  and JSON arguments;
- receive a typed `call_next` operation for the existing execution path;
- invoke `call_next` at most once;
- return either the real result, a replacement result, or a synthetic denied/error
  result.

When no interceptor is configured, `_execute_mcp_planned_tool_call()` will call the
existing path directly. Disabled mode must preserve current routing, results,
display, timing, and error behavior.

### Execution order

Transactional mode forces the planned calls in a model response through the existing
sequential runner. Only supported local coding tools enter the transactional
interceptor, but all calls in that response remain sequential so an unsupported call
cannot race a governed workspace mutation.

Baseline mode retains the current policy: more than one tool call may execute in
parallel unless fast-agent is globally configured to force sequential execution.

### Transaction semantics

The coordinator built on this boundary will order a supported call as:

```text
persist intent
  -> validate and authorize
  -> create a workspace checkpoint
  -> execute once
  -> persist the complete raw result as an artifact
  -> record committed or failed
  -> return a tool result to the model
```

The event store is the source of transaction facts. The artifact store owns complete
raw tool output. Existing session history remains the source for conversation
resume; neither store replaces session history, trajectory data, UI traces, or
provider history.

The project uses "transactional" to mean ordered, observable, and recoverable local
workspace operations. It does not claim database ACID guarantees for arbitrary tool
side effects.

## Guarantees

When transactional mode is enabled for a supported local tool:

- each call retains its model-provided correlation ID and receives a transaction ID;
- intent is durably recorded before the tool side effect begins;
- the complete raw result is stored before the transaction is committed;
- workspace writes execute serially;
- policy denial and execution failure produce an explicit tool result;
- recovery and final success are decided by runtime state and deterministic
  verification rather than by the model's completion claim.

When transactional mode is disabled, the existing fast-agent behavior remains the
compatibility baseline.

## Non-goals

The first phase does not:

- rewrite `ToolRunner`, the agent loop, Provider integration, SessionManager,
  compaction, usage tracking, permission handling, or execution environments;
- provide transactional governance for arbitrary remote MCP tools;
- provide automatic cross-process resume or resolve unknown outcomes after a crash;
- guarantee exactly-once execution for external effects;
- implement Saga compensation, parallel writes, path-level locking, or multi-agent
  transactions;
- guarantee rollback of files outside the run worktree, ignored files, nested
  repositories, submodules, dependency environments, or external systems;
- introduce a general policy language or a new long-term memory system.

## Rejected Alternatives

### Change the existing tool hooks to run once per call

Rejected because the hooks currently operate on request and result messages for a
whole batch. Changing that contract would affect all tool-loop users and still mix
transaction concerns into `ToolRunner`.

### Intercept every `McpAgent.call_tool()` invocation

Rejected for the first phase because `call_tool()` is also the routing boundary for
remote MCP and other non-coding tools. The planned-call boundary retains correlation
and execution metadata while allowing explicit local-tool eligibility.

### Add the contract to both `ToolAgent` and `McpAgent` immediately

Rejected because the two classes have separate planning and execution paths. A
shared general contract can be extracted later after the local `McpAgent` path has
proved the invariants.

### Preserve parallel execution for reads and independent writes

Rejected because the current parallel decision is based on call count, not
side-effect class or path conflicts. Safe read parallelism can be added only after
the runtime has explicit effect classification and conflict analysis.

## Consequences

- Transactional mode has additional persistence and checkpoint latency.
- The first implementation is deliberately specific to local coding tools.
- The per-call interceptor must remain optional and typed so it can later become a
  reusable execution contract.
- Domain events and transition rules, persistence stores, the interceptor, and the
  coordinator will be delivered separately so each invariant can be tested before
  the vertical path is enabled.
