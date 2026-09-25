# Declarations

The workflow language specification: immutable graph vocabulary plus the public
context, success, state, and callable interfaces used by generated-project code.

`interfaces.py` owns the generic feature and profile initializer protocols. Generated
`__init__.py` modules specialize them with project types and expose concrete forwarding functions.
The user-owned `impl.py` modules contain the bodies; imports are deferred until a
forwarding function is invoked.

Control ports are id-only markers and never execute or own state. Ordinary nodes
own one immutable local state value. Features own typed workflow-wide values
whose initializer may return `None`.
`WorkflowState` is an interpreter detail. Ordinary implementations receive only
their typed local `State` and `NodeContext`; a feature visit receives `FeatureState`,
which reads every slot but can replace only `FeatureDefinition` values.

An ordinary node declares only its shared output type and local-state type; the
state dataclass's default constructor supplies its initial value. Every incoming
edge owns one `VisitDefinition` or, for a durable call, one
`CallVisitDefinition`: its input type must equal that edge's source output type,
and its synchronous implementation is the overload run when that edge is
taken. Ordinary Python and Agent visits return the node's shared output and
next local state in `Success`. A Feature visit returns `FeatureSuccess`; a call
visit returns `Success` after its one durable child invocation. The runtime
builds the candidate workflow snapshot and validates it against the node
declaration. There is no node-level input contract or catch-all behavior
attached to the node.

Feature nodes likewise declare their shared output type but have no local state.
Their incoming-edge visits return `FeatureSuccess`, and the runtime forwards the
input payload unchanged after validating it against the node output type.

A subroutine is statically parameterized by `Input`, `Output`, and `Params` but
has no hidden executable boundary checks. Validation belongs in explicit graph
nodes and uses ordinary Python exceptions. A feature initializer is synchronous,
receives the subroutine input and Params, and returns its typed feature value or
`None`; its authored body is `initialize_impl`.

`NodeContext` identifies the run, graph, node, and incoming edge and exposes
`output_dir` as an absolute `pathlib.Path` dedicated to this visit. Node output
belongs there; the runtime does not change the process working directory around
node calls. Repeated visits receive different directories.

`AgentNodeContext` adds
`invoke(prompt, workspace=path, access=AgentAccess.READ_ONLY)`. The workspace is
required, write access is explicit, and an implementation may invoke its selected
profile and session zero or more times. Invocation is synchronous; the selected
`AgentInvoker` receives an `AgentRequest` and synchronously returns an
`AgentReply`. `CallContext` is the call API:
it adds one synchronous typed child invocation, including an explicit parameter
override, to the single implementation owned by `CallVisitDefinition`. Verdog
captures and replays that invocation for nested resumption, so the adapter is
deterministic and performs no unjournaled side effects.

Edges own the overload used to enter their target, but no state of their own.
They forward their source output, test conditions against the previous workflow
state, and test effects between the previous and candidate states. Exactly one
compatible outgoing edge commits the candidate. Edges targeting terminal ports
have no visit. Explicit effects are legal only on edges sourced by feature nodes;
omitted effects mean `UNCHANGED`.
Boolean, Integer, Float, and finite Enum features have checked runtime values.
Enum conditions require one declared value; enum effects require
one declared value or explicitly give up the constraint with `UNCONSTRAINED`.

A call visit's adapter invokes its child exactly once and maps the raw child
output into the parent node's `Success`; child state never crosses the boundary.
The adapter may be evaluated again after the child returns, while the child
execution and outcome are replayed rather than repeated. Both pre- and
post-invocation logic must therefore be deterministic and replay-safe.
In-process subroutine exceptions remain ordinary Python exceptions, so an adapter
may recover with `try`/`except` or let them propagate. An isolated workflow cannot
transport an exception object and raises `RemoteWorkflowError` with the remote
exception type and formatted traceback instead.
Both call kinds hold a definition id and owner-relative `project_path`.
`SubroutineCall` resolves the subroutine lazily and runs it in-process with the caller's
configuration. `WorkflowCall` resolves a workflow envelope in its isolated environment and
runs that envelope's referenced subroutine.
