# Verdog runtime

Execute [Verdog](https://drexlerd.github.io/verdog-website/) workflows in Python.
The runtime runs workflow steps and configured agents, records execution history,
and supports resuming or forking runs from compatible checkpoints.

Requires Python 3.12 or later.

## Installation

Generated workflow projects declare `verdog-runtime` as a dependency. To install
it directly in a Python environment:

```sh
uv pip install verdog-runtime
```

For creating, checking, and running projects from the terminal, install the
[Verdog CLI](https://github.com/verdog-ai/verdog-cli):

```sh
uv tool install verdog-cli
verdog --help
```

Workflow environments need the runtime and their declared dependencies. They do
not require the CLI or a backend installation to execute workflows.

## Documentation

- [Getting started](https://drexlerd.github.io/verdog-website/getting-started.html)
- [Running example](https://drexlerd.github.io/verdog-website/running-example.html)
- [Workflow declarations](verdog_runtime/declarations/README.md)
- [Interpreter reference](verdog_runtime/interpreter/README.md)
- [Execution logs and statistics](https://drexlerd.github.io/verdog-website/logging.html)

Runtime 0.1.6 stores checkpoint artifact inventories as add-only deltas and
shares their ancestry in memory. Existing full-inventory checkpoints remain
readable; existing run files are not rewritten.

Runtime 0.1.5 extends each `stats.md` timing table with agent call
counts, input and cached input tokens, output tokens, and estimated USD usage
cost. Child usage rolls into its parent call row; parent and previous/next call
links keep the existing report hierarchy navigable. Subscription cost estimates
describe usage, not additional charges or remaining quota. Token and cost cells
use `—` for Python, Feature, enter, exit, and failure rows where usage does not
apply. Agent, Subroutine call, Workflow call, and Total rows use `0` for no
usage, `unknown` for missing measurements, and a known subtotal followed by
`+ ?` for partial measurements. Agent call counts are always numeric.

## Development

See [developer documentation](DEVELOPMENT.md) for setup, checks, run-monitoring
APIs, and release instructions.

## License

AGPL-3.0-only; see [LICENSE](LICENSE).
