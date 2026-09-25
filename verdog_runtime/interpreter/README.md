# Interpreter

The reusable in-process interpreter for programs expressed with `declarations`.
It validates declarations, owns immutable per-run `WorkflowState`, dispatches
edge-selected node visits, and selects exactly one compatible outgoing edge at
each sequential step.

Python visits execute only when a generated project calls `Dispatcher`; the
catalogue backend never imports or executes project Python. Agent visits receive
an `AgentNodeContext` and may invoke a logical profile zero or more times.

Execution is synchronous and iterative. Exactly one `Dispatcher` owns an
interpreter process: its continuation stack, cancellation token, transition
budget, run store, invocation journal, parameters, statistics, and checkpoint
callbacks cover the root graph and every in-process subroutine activation.
Local calls into another physical project root remain frames on that stack and
retain their logical project path for durable addressing. A `WorkflowCall`
instead starts an isolated interpreter process, which has its own single
dispatcher and exposes only framed results and opaque checkpoint state to the
parent. No authored visit or local call requires a coroutine or recursively
grows the Python stack.

Each `WorkflowDefinition` carries a `WorkflowConfiguration` that binds the root
subroutine's profile parameters to `AgentInvoker` implementations. Subroutine
calls explicitly map their child's profile and session parameters to resources
in the caller.
There is no provider registry: the optional `CodexInvoker` and `ClaudeInvoker`
modules implement the same public protocol as a project-owned adapter. Persistent
sessions resume their provider conversation within one run; fresh sessions always
start fresh. A successful invocation advances the session immediately.

Every edge entering a computational node selects one visit overload. The visit
input type must exactly equal the source entity's declared output type (or the
subroutine input type when the source is `enter`). Before invocation the runtime
passes the forwarded value to that visit. The generated signature and local type
checker establish its static type. The visit returns the node's shared output with
its typed local state; the runtime validates the state against the node declaration.
There are no node-level input/output checks. A feature node has no local state: its selected visit forwards
its input and may propose feature changes by returning `FeatureSuccess` with a
candidate `FeatureState`.

At a subroutine boundary the interpreter initializes features in declaration
order from the input and then visits the enter port. An edge reaching the
marker-only failure port raises because the marker is terminal, not recovery code.
Failed initialization and implementation exceptions propagate as Python
exceptions. The interpreter adds graph and entity notes without replacing their
type, traceback, cause, or existing notes.

An edge's visit is executable only after the edge has been selected and its target
is entered. Edge conditions inspect the previous workflow state, and effects
compare it with a feature node's candidate state. The interpreter commits that
candidate only when exactly one outgoing edge is compatible; zero or multiple
matches raise. Only edges whose source is a feature node may declare effects;
omitted effects require the corresponding feature to remain unchanged.

Call visits use an explicit adapter to invoke their child exactly once and map its
result to the parent node declaration. Each child starts with its own workflow
state, which is never merged into the parent implicitly. A false condition or
effect makes an edge incompatible; invalid values, unauthorized writes, and
exceptions raise. `SubroutineCall` returns the child output directly and allows
ordinary in-process `try`/`except` recovery in its adapter. `WorkflowCall`
returns the remote output directly on success; a child exception is represented
as `RemoteWorkflowError`, carrying its qualified type and formatted remote
traceback. Cancelling either call still cleans up any subprocess it owns.

Each run writes beneath its selected output directory. Entity visits use
`graph-<subroutine-id>/<node-id>/<six-digit per-node counter>/`; each node starts at
`000001` and increments only when that node is revisited. The directory is created
and recorded in the newline-delimited `trace` file before the implementation runs.
Its lines retain the global visitation order, and revisits preserve earlier files,
including when a later visit fails.
Every exception crossing a graph boundary visits that graph's failure marker and
writes `stacktrace.txt` in the visit directory before re-raising. The file is
Python's native formatted traceback, including concise workflow context carried
by exception notes. Nested boundaries retain both the traceback and workflow
path without a parallel failure algebra.
Child call trees nest beneath their calling visit. The same layout crosses a
`WorkflowCall` process boundary, while each process keeps the working directory of
the project it owns rather than switching into an entity's output directory.

Every agent call receives
`<visit>/invocations/<six-digit invocation counter>/`. The built-in invokers store
the prompt, full event stream and stderr, the response on success, exposed
reasoning, and metadata there. These transport logs may contain sensitive project
content even though the invokers do not record the environment, credentials, or
full command line. The runtime supplies cancellation cleanup but no deadline,
retry, or fallback policy.

Dispatcher lifecycle methods block until completion or failure. A shared
`CancellationToken` may be cancelled by a supervising thread; provider and
child-process transports perform runtime-bounded blocking waits, cap them by
any caller-supplied deadline, and recheck the token between waits. A timeout
yields to the runtime rather than spinning. Managed subprocess trees are
terminated and reaped on cancellation. Ordinary authored Python remains
non-preemptible until it returns or raises. Durable checkpoint, session, and
retry semantics are documented in
[Runs and resumption](https://github.com/verdog-ai/verdog-website/blob/main/site/resumption.md).
