# Verdog runtime

The Python runtime and command-line client for [Verdog](https://github.com/verdog-ai/verdog-website).
It provides workflow declarations, the interpreter, agent adapters, durable run resumption,
and the `verdog` command. Project generation and verification use the hosted backend;
this package contains no backend or compiler implementation.
Requires Python 3.12 or newer.

Install or update the command-line client with Python 3.12 or newer:

```sh
uv tool install --upgrade 'verdog-runtime>=0.1.1'
verdog --help
```

The client uses `https://157.180.79.112` by default. Use
`verdog --backend-origin URL ...` to select another backend. Saved terminal sessions
retain the backend used when signing in.

Generated projects also declare `verdog-runtime` as a dependency. For local development,
install this checkout with `uv pip install .`.

## Development

```sh
uv sync --locked --no-editable
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync pytest
uv build
```

The tests and type-checking configuration are self-contained in this repository.
Use a regular installation (`--no-editable`) because `verdog sync` copies the installed
distribution into workflow environments.

### Code quality

Follow the [Google Python style guide](https://google.github.io/styleguide/pyguide.html).
Ruff enforces 80-column formatting, absolute imports, import sorting,
Google-style docstrings, and common correctness and simplification checks.
Import modules rather than their members, except for typing names and deliberate
public re-exports. Keep error messages, checkpoint formats, and public APIs stable
when refactoring. Documentation should explain contracts and non-obvious constraints.

Run `uv run --no-sync ruff format .` to format changes. Both pull requests and
releases check formatting, lint, types, and tests. The existing strict Pyright
configuration remains the authority for types; no type errors are suppressed to
satisfy a formatter.

For a deeper review, use the pinned analysis tools:

```sh
uv run --no-sync pylint verdog_runtime --reports=no
uv run --no-sync radon cc verdog_runtime -s -n C
uv run --no-sync radon mi verdog_runtime -s
```

Pylint and Radon are review aids, not score targets. Prioritize functions that
combine validation, mutation, and recovery; split them at those boundaries rather
than adding helpers just to meet a numeric threshold. Keep rollback and trust-boundary
checks intact. Cyclomatic complexity counts independent control-flow paths; inspect
both the largest function and any helpers extracted from it.

Compatibility exceptions are narrow: public package facades retain re-exports;
`TypeVar` and type-alias syntax remains where runtime introspection and checkpoint
compatibility depend on it; dynamic dataclass access uses `getattr` explicitly.
Tests use descriptive names and assertions instead of mandatory API docstrings.
Delayed imports in the CLI resolve its existing command/environment dependencies.

The September 2026 pass was measured against commit `e457564`, using the same
Ruff settings for both versions:

| Measure | Before | After |
| --- | ---: | ---: |
| Ruff diagnostics | 2,323 | 0 |
| Highest function cyclomatic complexity | 107 | 32 |
| Functions above complexity 20 | 11 | 9 |
| Mean function cyclomatic complexity | 4.64 | 4.56 |
| Function and nested-function count | 767 | 780 |

The largest changes separate dependency-removal validation from graph/file edits,
projection staging from rollback, and checkpoint validation from resource setup.
The extra functions represent those responsibilities; they do not change the
checkpoint or wire formats. Physical line counts rise with 80-column wrapping,
qualified names, and documentation, so line count alone is not the improvement
criterion.

See the [workflow declarations](verdog_runtime/declarations/README.md),
[interpreter](verdog_runtime/interpreter/README.md), and
[user documentation](https://github.com/verdog-ai/verdog-website/tree/main/site).

## Release

[release.yml](.github/workflows/release.yml) runs when a `v*` tag is pushed,
like `v0.1.1`. It checks that the tag matches `project.version`, runs style, tests, and
type checks on Python 3.12, builds the wheel and source distribution, checks
the installed wheel, and publishes both distributions to PyPI.

Configure this once before the first release:

1. Create the GitHub environment `pypi` in this repository's settings.
2. Add a [PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/):
   project `verdog-runtime`, owner `verdog-ai`, repository `verdog-runtime`,
   workflow filename `release.yml`, environment `pypi`.
   For a new PyPI project, add it as a pending publisher under your account's
   Publishing settings. For an existing project, use its Publishing settings.
   No API-token secret is needed.

Commit and push the repository contents, including this workflow. For each
release, update `project.version` in `pyproject.toml`, run `uv lock`, and commit
those changes. Then push the matching tag:

```sh
git push origin main
git tag v0.1.1
git push origin v0.1.1
```

Use a new version and matching tag for each subsequent release.

## License

Verdog runtime and its command-line client are licensed under the GNU Affero General Public License,
version 3 only (`AGPL-3.0-only`). See [LICENSE](LICENSE).
